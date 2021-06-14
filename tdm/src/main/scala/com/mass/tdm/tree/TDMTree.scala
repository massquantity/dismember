package com.mass.tdm.tree

import java.io._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import com.mass.tdm.protobuf.tree.Node

class TDMTree extends DistTree with Serializable {
  import DistTree.TreeNode

  override protected[tdm] val codeNodeMap = new mutable.HashMap[Int, Node]()
  override protected val idCodeMap = new mutable.HashMap[Int, Int]()
  override protected val leafCodes = new mutable.BitSet()
  override protected[tdm] var initialized: Boolean = false
  override protected[tdm] var maxLevel: Int = 0
  override protected[tdm] var maxCode: Int = -1
  override protected var nonLeafOffset: Int = -1

  private val paddingId: Int = 0  // padded original item id

  def getMaxLevel: Int = maxLevel

  def getIds: Array[Int] = idCodeMap.keys.toArray

  def getIdCodeMap: mutable.HashMap[Int, Int] = idCodeMap

  def init(path: String): Unit = {
    val (idCodeAllParts, treeMeta) = loadData(path)
    loadItems(idCodeAllParts, treeMeta)
    initialized = true
  }

  // paddingIndex = -1, paddingId = 0
  def idToCode(itemIds: Array[Int]): Array[Int] = {
    require(initialized, "tree hasn't been initialized...")
    val res = new Array[Int](itemIds.length)
    var i = 0
    while (i < itemIds.length) {
      val id = itemIds(i)
      if (id == paddingId) {
        res(i) = -1
      } else if (id < nonLeafOffset && idCodeMap.contains(id)) {
        res(i) = idCodeMap(id)
      } else {
     //   res(i) = id - nonLeafOffset
     //   if (res(i) > maxCode) res(i) = -1
        res(i) = -1
      }
      i += 1
    }
    res
  }

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

  def getAncestorNodes(itemCodes: Array[Int]): Array[Array[TreeNode]] = {
    itemCodes.map(code => {
      if (code <= 0 || !codeNodeMap.contains(code)) {
        Array.empty[TreeNode]
      } else {
        val res = new ArrayBuffer[TreeNode]()
        var _code = code
        while (_code != 0) {
          res += TreeNode(_code, codeNodeMap(_code))
          _code = (_code - 1) >> 1
        }
        res.toArray
      }
    })
  }

  def getChildNodes(itemCode: Int): List[TreeNode] = {
    var res = List.empty[TreeNode]
    val childLeft = 2 * itemCode + 1
    if (codeNodeMap.contains(childLeft)) {
      res = TreeNode(childLeft, codeNodeMap(childLeft)) :: res
    }
    val childRight = 2 * itemCode + 2
    if (codeNodeMap.contains(childRight)) {
      res = TreeNode(childRight, codeNodeMap(childRight)) :: res
    }
    res
  }
}

object TDMTree {

  def apply(pbFilePath: String): TDMTree = {
    val tree = new TDMTree
    tree.init(pbFilePath)
    tree
  }
}
