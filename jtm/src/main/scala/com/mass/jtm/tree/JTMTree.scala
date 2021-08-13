package com.mass.jtm.tree

import java.io.{BufferedOutputStream, FileNotFoundException, OutputStream}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.{Failure, Success, Using}

import com.mass.sparkdl.utils.{FileWriter => DistFileWriter}
import com.mass.tdm.protobuf.store_kv.KVItem
import com.mass.tdm.protobuf.tree.{IdCodePair, IdCodePart, Node, TreeMeta}
import com.mass.tdm.tree.DistTree
import com.mass.tdm.tree.TreeBuilder.{toByteString, writeKV}
import org.apache.log4j.Logger

class JTMTree extends DistTree with Serializable {
  import JTMTree._
  val logger: Logger = Logger.getLogger(getClass)

  override protected[jtm] val codeNodeMap = mutable.Map.empty[Int, Node]
  override protected val idCodeMap = mutable.Map.empty[Int, Int]
  override protected[jtm] val leafCodes = new mutable.BitSet()
  override protected[jtm] var initialized: Boolean = false
  override protected[jtm] var maxLevel: Int = 0
  override protected[jtm] var maxCode: Int = -1
  override protected var nonLeafOffset: Int = -1

  private val paddingId: Int = 0  // padded original item id

  def init(path: String): Unit = {
    val (idCodeAllParts, treeMeta) = loadData(path)
    loadItems(idCodeAllParts, treeMeta)
    initialized = true
  }

  def getAncestorAtLevel(item: Int, level: Int): Int = {
    val maxCode = math.pow(2, level + 1).toInt - 1
    var code = idCodeMap(item)
    while (code >= maxCode) {
      code = (code - 1) >> 1
    }
    code
  }

  def getAllNodesAtLevel(level: Int): Array[Int] = {
    val levelStart = math.pow(2, level).toInt - 1
    val levelEnd = levelStart * 2 + 1
    (levelStart until levelEnd).toArray.filter(codeNodeMap.contains)
  }

  def getChildrenAtLevel(ancestor: Int, oldLevel: Int, level: Int): Array[Int] = {
    var diff = level - oldLevel
    var parent = Array(ancestor)
    val children = ArrayBuffer.empty[Int]
    var finished = false
    while (!finished) {
      parent.foreach { p =>
        children += 2 * p + 1
        children += 2 * p + 2
      }
      diff -= 1
      if (diff == 0) {
        finished = true
      } else {
        parent = children.toArray
        children.clear()
      }
    }

    children.toArray.filter(codeNodeMap.contains)  // todo: children at different places
}

  def idToCode(
      itemIds: Array[Int],
      level: Int,
      hierarchical: Boolean,
      minLevel: Int): Array[Int] = {

    val res = new Array[Int](itemIds.length)
    var i = 0
    while (i < itemIds.length) {
      val id = itemIds(i)
      if (id == paddingId) {
        res(i) = -1
      } else if (id < nonLeafOffset && idCodeMap.contains(id)) {
        res(i) =
          if (hierarchical && level >= minLevel) {
            getAncestorAtLevel(id, level)
          } else {
            idCodeMap(id)
          }
      } else {
        res(i) = id - nonLeafOffset
        if (res(i) > maxCode) res(i) = -1
      }
      i += 1
    }
    res
  }

  def idToCodeWithMask(
      itemIds: Array[Int],
      level: Int,
      hierarchical: Boolean,
      minLevel: Int): (Array[Int], ArrayBuffer[Int]) = {

    val mask = new ArrayBuffer[Int]()
    val res = new Array[Int](itemIds.length)
    var i = 0
    while (i < itemIds.length) {
      val id = itemIds(i)
      if (id == paddingId) {
        res(i) = -1
        mask += i
      } else if (id < nonLeafOffset && idCodeMap.contains(id)) {
        res(i) =
          if (hierarchical && level >= minLevel) {
            getAncestorAtLevel(id, level)
          } else {
            idCodeMap(id)
          }
      } else {
        res(i) = id - nonLeafOffset
        if (res(i) > maxCode) res(i) = -1
      }
      i += 1
    }
    (res, mask)
  }

  def flattenLeaves(projectionPi: mutable.Map[Int, Int], maxLevel: Int): Unit = {
    val minLeafCode = math.pow(2, maxLevel).toInt - 1
    val projection = projectionPi.toArray
    val projectionLeafCodes = projection.map(_._2).filter(_ >= minLeafCode).toSet
    val unassignedLeafCodes = leafCodes.diff(projectionLeafCodes)
    val (noPlaceItems, originalPlaceItems) = projection
      .filter(_._2 < minLeafCode)
      .partition(i => {
        val oldCode = idCodeMap(i._1)
        projectionLeafCodes.contains(oldCode)
      })

    println(s"unassigned nodes: ${unassignedLeafCodes.size}, " +
      s"no place items: ${noPlaceItems.length}, " +
      s"original place items: ${originalPlaceItems.length}")

    // projectionPi doesn't contain these leaf codes, so original places are kept
    originalPlaceItems.foreach { case (itemId, code) =>
      val oldCode = idCodeMap(itemId)
      projectionPi(itemId) = oldCode
      unassignedLeafCodes -= oldCode
    }

    // assign to nearest leaf code
    noPlaceItems.foreach { case (itemId, code) =>
      val leafCode = code * 2 + 1
      val nearestCode = unassignedLeafCodes.reduce { (a, b) =>
        if (math.abs(a - leafCode) < math.abs(b - leafCode)) a else b
      }
      projectionPi(itemId) = nearestCode
      unassignedLeafCodes -= nearestCode
    }

    println("unassigned nodes remained: " + unassignedLeafCodes.size)
    require(unassignedLeafCodes.isEmpty, "still remains unassigned codes")
  }

  def writeTree(projectionPi: mutable.Map[Int, Int], pbFilePath: String): Unit = {
    val leafStat = mutable.Map.empty[Int, Float]
    val pstat = mutable.Map.empty[Int, Float]
    // some original leaf nodes may stay in upper level, so put all of them to leaves
    flattenLeaves(projectionPi, maxLevel - 1)

    projectionPi.foreach { case (itemId, newCode) =>
      val oldCode = idCodeMap(itemId)
      val prob = codeNodeMap(oldCode).probality
      leafStat(newCode) = prob
      val ancestors = getAncestors(newCode)
      ancestors.foreach { anc =>
        pstat(anc) = pstat.getOrElse(anc, 0.0f) + prob
      }
    }

    val fileWriter = DistFileWriter(pbFilePath)
    val output: OutputStream = fileWriter.create(overwrite = true)
    Using(new BufferedOutputStream(output)) { writer =>
      val parts = new ArrayBuffer[IdCodePart]()
      val tmpItems = new ArrayBuffer[IdCodePair]()
      val savedNodes = new mutable.BitSet()
      val itemIds = projectionPi.keys.toArray
      var i = 0
      while (i < itemIds.length) {
        val id = itemIds(i)
        val code = projectionPi(id)
        val prob = leafStat(code)
        val leafCatId = 0
        val isLeaf = true
        val leafNode = Node(id, prob, leafCatId, isLeaf)
        val leafKV = KVItem(toByteString(code), leafNode.toByteString)
        writeKV(leafKV, writer)

        tmpItems += IdCodePair(id, code)
        if (i == itemIds.length - 1 || tmpItems.length == 512) {
          val partId = "Part_" + (parts.length + 1)
          parts += IdCodePart(toByteString(partId), tmpItems.clone())
          tmpItems.clear()
        }

        val ancestors = getAncestors(code)
        ancestors.foreach { ancCode =>
          if (!savedNodes.contains(ancCode)) {
            val id = ancCode + nonLeafOffset
            val prob = pstat(ancCode)
            val leafCatId = 0
            val isLeaf = false
            val node = Node(id, prob, leafCatId, isLeaf)
            val ancestorKV = KVItem(toByteString(ancCode), node.toByteString)
            writeKV(ancestorKV, writer)
            savedNodes.add(ancCode)
          }
        }
        i += 1
      }

      parts.foreach { p =>
        val partKV = KVItem(p.partId, p.toByteString)
        writeKV(partKV, writer)
      }

      val partIds = parts.map(x => x.partId).toArray
      val meta = TreeMeta(maxLevel, partIds)
      val metaKV = KVItem(toByteString("tree_meta"), meta.toByteString)
      writeKV(metaKV, writer)

    } match {
      case Success(_) =>
        output.close()
        fileWriter.close()
      case Failure(e: FileNotFoundException) =>
        println(s"""file "$pbFilePath" not found""")
        throw e
      case Failure(t: Throwable) =>
        throw t
    }

    logger.info(s"item num: ${projectionPi.size}, " +
      s"tree level: $maxLevel, " +
      s"leaf code start: ${projectionPi.values.min}, " +
      s"leaf code end: ${projectionPi.values.max}")
  }
}

object JTMTree {

  def apply(pbFilePath: String): JTMTree = {
    val tree = new JTMTree
    tree.init(pbFilePath)
    tree
  }

  def getAncestors(code: Int): Array[Int] = {
    val ancs = new ArrayBuffer[Int]()
    var num = code
    while (num > 0) {
      num = (num - 1) / 2
      ancs += num
    }
    ancs.toArray
  }

  /*
  def getChildrenAtLevel(ancestor: Int, oldLevel: Int, level: Int): Array[Int] = {
  //  val levelStart = math.pow(2, level).toInt - 1
  //  val levelEnd = levelStart * 2 + 1
    var diff = level - oldLevel
    var parent = Array(ancestor)
    val children = ArrayBuffer.empty[Int]
    // index start from 0, so all level subtract 1
    var _level = oldLevel
    val _maxLevel = maxLevel - 1
    var finished = false
    while (!finished) {
      if (_level < _maxLevel) {
        parent.foreach { p =>
          val childLeft = 2 * p + 1
          if (codeNodeMap.contains(childLeft)) {
            children += childLeft
          }
          val childRight = 2 * p + 2
          if (codeNodeMap.contains(childRight)) {
            children += childRight
          }
        }
      } else {
        // leaf nodes can be different in final tree, so didn't exclude them
        parent.foreach { p =>
          children += 2 * p + 1
          children += 2 * p + 2
        }
      }

      _level += 1
      diff -= 1
      if (diff == 0) {
        finished = true
      } else {
        parent = children.toArray
        children.clear()
      }
      /*
      if (children.head >= levelStart && children.head < levelEnd) {
        finished = true
      } else {
        parent = children.toArray
        children.clear()
      } */
    }
    children.toArray
  } */

  /*
  Using(new BufferedOutputStream(output)) { writer =>
    val parts = new ArrayBuffer[IdCodePart]()
    val tmpItems = new ArrayBuffer[IdCodePair]()
    val codes = codeNodeMap.keys.toArray
    var i = 0
    while (i < codes.length) {
      val code = codes(i)
      if (leafStat.contains(code)) {
        val id = leafStat(code)._1
        val prob = leafStat(code)._2
        val leafCatId = 0
        val isLeaf = true
        val leafNode = Node(id, prob, leafCatId, isLeaf)
        val leafKV = KVItem(toByteString(code), leafNode.toByteString)
        writeKV(leafKV, writer)

        tmpItems += IdCodePair(id, code)
        if (i == codes.length - 1 || tmpItems.length == 512) {
          val partId = "Part_" + (parts.length + 1)
          parts += IdCodePart(toByteString(partId), tmpItems.clone())
          tmpItems.clear()
        }
      } else {
        val id = codeNodeMap(code).id
        val prob = pstat(code)
        val leafCatId = 0
        val isLeaf = false
        val node = Node(id, prob, leafCatId, isLeaf)
        val ancestorKV = KVItem(toByteString(code), node.toByteString)
        writeKV(ancestorKV, writer)
      }
      i += 1
    }

    parts.foreach { p =>
      val partKV = KVItem(p.partId, p.toByteString)
      writeKV(partKV, writer)
    }

    val partIds = parts.map(x => x.partId).toArray
    val meta = TreeMeta(maxLevel, partIds)
    val metaKV = KVItem(toByteString("tree_meta"), meta.toByteString)
    writeKV(metaKV, writer)

  } match {
    case Success(_) =>
      output.close()
      fileWriter.close()
    case Failure(e: FileNotFoundException) =>
      println(s"""file "$pbFilePath" not found""")
      throw e
    case Failure(t: Throwable) =>
      throw t
  }

  logger.info(s"item num: ${projectionPi.size}, " +
    s"tree level: $maxLevel, " +
    s"leaf code start: ${projectionPi.values.min}, " +
    s"leaf code end: ${projectionPi.values.max}")
   */

}
