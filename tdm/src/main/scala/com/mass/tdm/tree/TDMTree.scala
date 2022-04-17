package com.mass.tdm.tree

import java.io._

import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import com.mass.tdm.{paddingId, paddingIdx}
import com.mass.tdm.protobuf.tree.Node

class TDMTree extends DistTree with Serializable {
  import DistTree.TreeNode

  override protected[tdm] val codeNodeMap = mutable.Map.empty[Int, Node]
  override protected val idCodeMap = mutable.Map.empty[Int, Int]
  override protected val leafCodes = new mutable.BitSet()
  override protected[tdm] var initialized: Boolean = false
  override protected[tdm] var maxLevel: Int = 0
  override protected[tdm] var maxCode: Int = -1
  override protected var nonLeafOffset: Int = -1

  def getMaxLevel: Int = maxLevel

  def getIds: Array[Int] = idCodeMap.keys.toArray

  def getIdCodeMap: mutable.Map[Int, Int] = idCodeMap

  def init(path: String): Unit = {
    val (idCodeAllParts, treeMeta) = loadData(path)
    loadItems(idCodeAllParts, treeMeta)
    initialized = true
  }

  def idToCode(itemIds: Array[Int]): (Seq[Int], Array[Int]) = {
    val masks = new ArrayBuffer[Int]()
    val codes = itemIds.zipWithIndex.map { case (id, i) =>
      if (id == paddingId) {
        masks += i
        paddingIdx
      } else if (id < nonLeafOffset && idCodeMap.contains(id)) {
        // leaves
        idCodeMap(id)
      } else {
        // ancestors
        val tmp = id - nonLeafOffset
        if (tmp > maxCode) {
          masks += i
          paddingIdx
        } else {
          tmp
        }
      }
    }
    (codes.toSeq, masks.toArray)
  }

  // get current node and its ancestors, but exclude root node
  def pathNodes(itemCodes: Seq[Int]): Seq[List[TreeNode]] = {
    @tailrec
    def upTrace(code: Int, res: List[TreeNode]): List[TreeNode] = {
      if (code == 0) {
        res
      } else {
        val parent = (code - 1) >> 1
        val node = TreeNode(code, codeNodeMap(code))
        upTrace(parent, node :: res)
      }
    }

    itemCodes map { code =>
      if (code <= 0 || !codeNodeMap.contains(code)) {
        Nil
      } else {
        upTrace(code, Nil)
      }
    }
  }
}

object TDMTree {

  def apply(pbFilePath: String): TDMTree = {
    val tree = new TDMTree
    tree.init(pbFilePath)
    tree
  }
}
