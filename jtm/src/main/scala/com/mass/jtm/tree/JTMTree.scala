package com.mass.jtm.tree

import java.io.{BufferedOutputStream, OutputStream}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Using

import com.mass.sparkdl.utils.{FileWriter => DistFileWriter}
import com.mass.tdm.{paddingId, paddingIdx}
import com.mass.tdm.protobuf.store_kv.KVItem
import com.mass.tdm.protobuf.tree.{IdCodePair, IdCodePart, Node, TreeMeta}
import com.mass.tdm.tree.DistTree
import com.mass.tdm.tree.TreeBuilder.{toByteString, writeKV}
import org.apache.log4j.Logger

class JTMTree extends DistTree with Serializable {
  import JTMTree._
  val logger: Logger = Logger.getLogger(getClass)

  override protected[jtm] val codeNodeMap = mutable.Map.empty[Int, Node]
  override protected[jtm] val idCodeMap = mutable.Map.empty[Int, Int]
  override protected[jtm] val leafCodes = new mutable.BitSet()
  override protected[jtm] var initialized: Boolean = false
  override protected[jtm] var maxLevel: Int = 0
  override protected[jtm] var maxCode: Int = -1
  override protected var nonLeafOffset: Int = -1

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

  // filter(codeNodeMap.contains)
  def getAllNodesAtLevel(level: Int): Array[Int] = {
    val levelStart = math.pow(2, level).toInt - 1
    val levelEnd = levelStart * 2 + 1
    Array.range(levelStart, levelEnd)
  }

  // filter(codeNodeMap.contains)  explore different nodes
  def getChildrenAtLevel(ancestor: Int, oldLevel: Int, level: Int): Array[Int] = {
    (oldLevel until level).foldLeft(Array(ancestor)) { case (nodes, _) =>
      nodes.flatMap(n => Array(n * 2 + 1, n * 2 + 2))
    }
  }

  def idToCode(
    itemIds: Array[Int],
    level: Int,
    hierarchical: Boolean,
    minLevel: Int
  ): Array[Int] = {
    val res = new Array[Int](itemIds.length)
    var i = 0
    while (i < itemIds.length) {
      val id = itemIds(i)
      if (id == paddingId) {
        res(i) = paddingIdx
      } else if (id < nonLeafOffset && idCodeMap.contains(id)) {
        res(i) =
          if (hierarchical && level >= minLevel) {
            getAncestorAtLevel(id, level)
          } else {
            idCodeMap(id)
          }
      } else {
        res(i) = id - nonLeafOffset
        if (res(i) > maxCode) res(i) = paddingIdx
      }
      i += 1
    }
    res
  }

  def idToCodeWithMask(
    itemIds: Array[Int],
    level: Int,
    hierarchical: Boolean,
    minLevel: Int
  ): (Array[Int], ArrayBuffer[Int]) = {
    val mask = new ArrayBuffer[Int]()
    val res = new Array[Int](itemIds.length)
    var i = 0
    while (i < itemIds.length) {
      val id = itemIds(i)
      if (id == paddingId) {
        res(i) = paddingIdx
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
        if (res(i) > maxCode) res(i) = paddingIdx
      }
      i += 1
    }
    (res, mask)
  }

  def writeTree(projectionPi: Map[Int, Int], pbFilePath: String): Unit = {
    // some original leaf nodes may stay in upper level, so put all of them to leaves
    // flattenLeaves(projectionPi, maxLevel)
    val leafStat = mutable.Map.empty[Int, Float]
    val pstat = mutable.Map.empty[Int, Float]
    projectionPi.foreach { case (itemId, newCode) =>
      val oldCode = idCodeMap(itemId)
      val prob = codeNodeMap(oldCode).probality
      leafStat(newCode) = prob
      val ancestors = getAncestors(newCode, maxLevel)
      ancestors.foreach { anc =>
        pstat(anc) = pstat.getOrElse(anc, 0.0f) + prob
      }
    }

    val fileWriter = DistFileWriter(pbFilePath)
    val output: OutputStream = fileWriter.create(overwrite = true)
    Using.resource(new BufferedOutputStream(output)) { writer =>
      val parts = new ArrayBuffer[IdCodePart]()
      val tmpItems = new ArrayBuffer[IdCodePair]()
      val savedNodes = new mutable.BitSet()
      projectionPi.zipWithIndex.foreach { case ((id, code), i) =>
        val prob = leafStat(code)
        val leafCatId = 0
        val isLeaf = true
        val leafNode = Node(id, prob, leafCatId, isLeaf)
        val leafKV = KVItem(toByteString(code), leafNode.toByteString)
        writeKV(leafKV, writer)

        tmpItems += IdCodePair(id, code)
        if (i == projectionPi.size - 1 || tmpItems.length == 512) {
          val partId = "Part_" + (parts.length + 1)
          parts += IdCodePart(toByteString(partId), tmpItems.clone().toSeq)
          tmpItems.clear()
        }

        val ancestors = getAncestors(code, maxLevel)
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
      }

      parts.foreach { p =>
        val partKV = KVItem(p.partId, p.toByteString)
        writeKV(partKV, writer)
      }
      val partIds = parts.map(x => x.partId).toArray
      val meta = TreeMeta(maxLevel, partIds)
      val metaKV = KVItem(toByteString("tree_meta"), meta.toByteString)
      writeKV(metaKV, writer)
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

  def getAncestors(code: Int, maxLevel: Int): List[Int] = {
    val ancestors = List.fill(maxLevel)(0)
    ancestors.scanLeft(code)((a, _) => (a - 1) / 2).tail
  }
}
