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

  override protected[jtm] val codeNodeMap = new mutable.HashMap[Int, Node]()
  override protected val idCodeMap = new mutable.HashMap[Int, Int]()
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
    val maxCode = (math.pow(2, level + 1) - 1).toInt
    var code = idCodeMap(item)
    while (code >= maxCode) {
      code = (code - 1) >> 1
    }
    code
  }

  def getAllNodesAtLevel(level: Int): Array[Int] = {
    val levelStart = (math.pow(2, level) - 1).toInt
    val levelEnd = levelStart * 2 + 1
    (levelStart until levelEnd).toArray.filter(codeNodeMap.contains)
  }

  def getChildrenAtLevel(ancestor: Int, level: Int): Array[Int] = {
    val levelStart = (math.pow(2, level) - 1).toInt
    val levelEnd = levelStart * 2 + 1
    var parent = Array(ancestor)
    val children = ArrayBuffer.empty[Int]
    // index start from 0, so all level subtract 1
    val _maxLevel = maxLevel - 1
    var finished = false
    while (!finished) {
      if (level < _maxLevel) {
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

      if (children.head >= levelStart && children.head < levelEnd) {
        finished = true
      } else {
        parent = children.toArray
        children.clear()
      }
    }
    children.toArray
  }

  // paddingIndex = -1, paddingId = 0
  def idToCodeWithMask(itemIds: Array[Int]): (Array[Int], ArrayBuffer[Int]) = {
    val mask = new ArrayBuffer[Int]()
    val res = new Array[Int](itemIds.length)
    var i = 0
    while (i < itemIds.length) {
      val id = itemIds(i)
      if (id == paddingId) {
        res(i) = -1
        mask += i
      } else if (id < nonLeafOffset && idCodeMap.contains(id)) {
        res(i) = idCodeMap(id)
      } else {
      //  res(i) = id - nonLeafOffset
      //  if (res(i) > maxCode) res(i) = -1
        res(i) = -1
      }
      i += 1
    }
    (res, mask)
  }

  def writeTree(projectionPi: mutable.HashMap[Int, Int], pbFilePath: String): Unit = {
    val leafStat = mutable.HashMap.empty[Int, (Int, Float)]
    val pstat = mutable.HashMap.empty[Int, Float]

    projectionPi.foreach { case (itemId, newCode) =>
      val oldCode = idCodeMap(itemId)
      val prob = codeNodeMap(oldCode).probality
      leafStat(newCode) = (itemId, prob)
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
      val leafCodes = leafStat.keys.toArray
      val ancestorCodes = pstat.keys.toArray

      leafCodes.foreach { code =>
        val id = leafStat(code)._1
        val prob = leafStat(code)._2
        val leafCatId = 0
        val isLeaf = true
        val leafNode = Node(id, prob, leafCatId, isLeaf)
        val leafKV = KVItem(toByteString(code), leafNode.toByteString)
        writeKV(leafKV, writer)

        tmpItems += IdCodePair(id, code)
        if (code == leafCodes.last || tmpItems.length == 512) {
          val partId = "Part_" + (parts.length + 1)
          parts += IdCodePart(toByteString(partId), tmpItems.clone())
          tmpItems.clear()
        }
      }

      ancestorCodes.foreach { code =>
        val id = codeNodeMap(code).id
        val prob = pstat(code)
        val leafCatId = 0
        val isLeaf = false
        val node = Node(id, prob, leafCatId, isLeaf)
        val ancestorKV = KVItem(toByteString(code), node.toByteString)
        writeKV(ancestorKV, writer)
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
  def getChildrenAtLevel(ancestor: Int, level: Int): Array[Int] = {
    val levelStart = (math.pow(2, level) - 1).toInt
    val levelEnd = levelStart * 2 + 1
    var parent = Array(ancestor)
    val children = ArrayBuffer.empty[Int]
    var finished = false
    while (!finished) {
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

      newChildren = {
        if (children.count(codeNodeMap.contains) <= 1) {
          children.toArray
        } else {
          children.filter(codeNodeMap.contains).toArray
        }
      }

      if (children.isEmpty || (children.head >= levelStart && children.head < levelEnd)) {
        finished = true
      } else {
        parent = children.toArray
        children.clear()
      }
    }
    children.toArray
  }
 */

}
