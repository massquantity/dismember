package com.mass.tdm.model

import scala.collection.mutable.ArrayBuffer
import scala.math.Ordering

import com.mass.sparkdl.Module
import com.mass.sparkdl.nn.abstractnn.Activity
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.utils.{T, Table}
import com.mass.tdm.protobuf.tree.Node
import com.mass.tdm.tree.TDMTree
import com.mass.tdm.tree.DistTree.TreeNode

trait Recommender {
  import Recommender.{TreeNodePred, duplicateSequence}

  def recommendItems(
      sequence: Array[Int],
      model: Module[Float],
      tree: TDMTree,
      topk: Int,
      candidateNum: Int,
      useMask: Boolean): Array[Int] = {

    val recs = _recommend(sequence, model, tree, candidateNum, useMask)
    // recs.sorted(Ordering.by[TreeNodePred, Float](_.pred)(Ordering[Float].reverse))
    recs.sortBy(_._2)(Ordering[Float].reverse).take(topk).map(_._1)
  }

  private[model] def _recommend(
      sequence: Array[Int],
      model: Module[Float],
      tree: TDMTree,
      candidateNum: Int,
      useMask: Boolean): Array[(Int, Float)] = {

    // binary tree with 2 child => 2 * candidateNum
    val modelInputs = duplicateSequence(sequence, tree, useMask, candidateNum * 2)
    val leafIds = ArrayBuffer.empty[(Int, Float)]
    val childNodes = ArrayBuffer.empty[TreeNode]
    val candidateNodes = ArrayBuffer.empty[TreeNodePred]

    // no need to compute score if nodes number < candidateNum,
    // initialize candidateNodes to number of candidateNum
    var i = getLevelStart(candidateNum)
    val levelEnd = i * 2 + 1
    while (tree.codeNodeMap.contains(i) && i < levelEnd) {
      candidateNodes += TreeNodePred(i, tree.codeNodeMap(i), 0.0f)
      i += 1
    }

    while (candidateNodes.nonEmpty) {
      var (leafNodes, nonLeafNodes) = candidateNodes.partition(
        n => tree.codeNodeMap(n.code).isLeaf)

      if (leafNodes.nonEmpty) {
        leafNodes.foreach(i => {
          leafIds += Tuple2(i.node.id, i.pred)
        })
      }

      if (nonLeafNodes.isEmpty) {
        candidateNodes.clear()
      } else {
        if (nonLeafNodes.length > candidateNum) {
          nonLeafNodes = nonLeafNodes.sorted(new Ordering[TreeNodePred] {
            override def compare(x: TreeNodePred, y: TreeNodePred): Int = {
              y.pred.compareTo(x.pred)
            }
          })
        }

        var i = 0
        while (i < nonLeafNodes.length && i < candidateNum) {
          val internalNode = nonLeafNodes(i)
          val children = tree.getChildNodes(internalNode.code)
          children.foreach(c => childNodes += c)
          i += 1
        }

        val feature = modelInputs.generateInputs(childNodes)
        val preds = model.forward(feature).asInstanceOf[Tensor[Float]].storage().array()
        candidateNodes.clear()

        i = 0
        val nextLen = childNodes.length
        while (i < nextLen) {
          candidateNodes += TreeNodePred(childNodes(i).code, childNodes(i).node, preds(i))
          i += 1
        }
        childNodes.clear()
      }
    }

    leafIds.toArray
  }

  @inline
  def sigmoid(logit: Float): Double = {
    1.0 / (1 + java.lang.Math.exp(-logit))
  }

  def getLevelStart(candidateNum: Int): Int = {
    var i = 0
    var _can = candidateNum
    while (_can > 0) {
      i = i * 2 + 1
      _can  = (_can - 1) / 2
    }
    i
  }
}

object Recommender {

  case class TreeNodePred(code: Int, node: Node, pred: Float)

  private class ModelInputs(
      targetItem: Tensor[Int],
      seqItems: Tensor[Int],
      masks: Tensor[Int] = null,
      maskLen: Int = -1,
      useMask: Boolean) {

    def generateInputs(targetNodes: ArrayBuffer[TreeNode]): Table = {
      val num = targetNodes.length
      var i = 0
      while (i < num) {
        targetItem.setValue(i, 0, targetNodes(i).code)
        i += 1
      }
      val item = targetItem.narrow(0, 0, num)
      val seq = seqItems.narrow(0, 0, num)
      if (useMask) {
        // masks is one-dimensional tensor
        val mask = if (masks.isEmpty) masks else masks.narrow(0, 0, num * maskLen)
        T(item, seq, mask)
      } else {
        T(item, seq)
      }
    }
  }

  private def duplicateSequence(sequence: Array[Int], tree: TDMTree,
      useMask: Boolean, candidateLen: Int): ModelInputs = {

    val seqLen = sequence.length
    val features = new Array[Int](candidateLen * seqLen)
    val targets = new Array[Int](candidateLen)
    val targetItems = Tensor(targets, Array(candidateLen, 1))
    val seqItems = Tensor(features, Array(candidateLen, seqLen))

    if (!useMask) {
      val seqCodes = tree.idToCode(sequence)
      var i = 0
      while (i < candidateLen) {
        val offset = i * seqLen
        System.arraycopy(seqCodes, 0, features, offset, seqLen)
        //  features(offset + seqLen) = candidate(i).code
        i += 1
      }

      new ModelInputs(targetItem = targetItems, seqItems = seqItems,
        useMask = false)

    } else {
      val (seqCodes, mask) = tree.idToCodeWithMask(sequence)
      val maskBuffer = new ArrayBuffer[Int]()
      var i = 0
      while (i < candidateLen) {
        val offset = i * seqLen
        System.arraycopy(seqCodes, 0, features, offset, seqLen)
        // targets(i) = candidate(i).code
        mask.foreach(m => maskBuffer += (m + offset))
        i += 1
      }

      val masks = if (maskBuffer.isEmpty) {
        Tensor[Int]()
      } else {
        Tensor(maskBuffer.toArray, Array(maskBuffer.length))
      }

      new ModelInputs(targetItem = targetItems, seqItems = seqItems,
        masks = masks, maskLen = mask.length, useMask = true)
    }
  }
}
