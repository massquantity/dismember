package com.mass.tdm.model

import scala.collection.mutable.ArrayBuffer
import scala.math.Ordering

import com.mass.sparkdl.Module
import com.mass.sparkdl.tensor.Tensor
import com.mass.tdm.protobuf.tree.Node
import com.mass.tdm.tree.DistTree
import com.mass.tdm.tree.DistTree.TreeNode

trait Recommender {
  import Recommender.{TreeNodePred, duplicateSequence}

  def recommendItems(
      sequence: Array[Int],
      model: Module[Float],
      tree: DistTree,
      topk: Int,
      candidateNum: Int): Array[Int] = {

    val recs = _recommend(sequence, model, tree, candidateNum)
    // recs.sorted(Ordering.by[TreeNodePred, Float](_.pred)(Ordering[Float].reverse))
    recs.sortBy(_._2)(Ordering[Float].reverse).take(topk).map(_._1)
  }

  private[model] def _recommend(
      sequence: Array[Int],
      model: Module[Float],
      tree: DistTree,
      candidateNum: Int): Array[(Int, Float)] = {

    val seqCodes = tree.idToCode(sequence)
    val leafIds = ArrayBuffer.empty[(Int, Float)]
    val childNodes = ArrayBuffer.empty[TreeNode]
    val rootNode = TreeNodePred(0, tree.kvData(0.toString), 0.0f)
    val candidateNodes: ArrayBuffer[TreeNodePred] = ArrayBuffer(rootNode)

    while (candidateNodes.nonEmpty) {
      var (leafNodes, nonLeafNodes) = candidateNodes.partition(
        n => tree.getChildNodes(n.code).isEmpty)

      if (leafNodes.nonEmpty) {
        leafNodes.foreach(i => {
          val node = Node.parseFrom(i.rawNode)
          leafIds += Tuple2(node.id, i.pred)
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
          childNodes ++= tree.getChildNodes(internalNode.code)
          i += 1
        }

        val feature = duplicateSequence(seqCodes, childNodes)
        val preds = model.forward(feature).asInstanceOf[Tensor[Float]].storage().array()
        candidateNodes.clear()

        childNodes.zip(preds) map { case (c, p) =>
          candidateNodes += TreeNodePred(c.code, c.node, p)
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
}

object Recommender {

  case class TreeNodePred(code: Int, rawNode: Array[Byte], pred: Float)

  private def duplicateSequence(sequence: Seq[Int], candidate: ArrayBuffer[TreeNode]): Tensor[Int] = {
    val seqLen = sequence.length
    val newLen = seqLen + 1
    val canLen = candidate.length
    val features = Array.fill[Int](newLen * canLen)(0)
    var i = 0
    while (i < canLen) {
      val offset = i * newLen
      var j = 0
      while (j < seqLen) {
        features(offset + j) = sequence(j)
        j += 1
      }
      features(offset + j) = candidate(i).code
      i += 1
    }
    Tensor(features, Array(canLen, newLen))
  }

  def recommendWithProb2(
      model: Module[Float],
      tree: DistTree,
      topk: Int,
      candidateNum: Int,
      sequence: Array[Int],
      transform: (Array[Int], ArrayBuffer[TreeNode]) => Tensor[Int]): Unit = {

    val seqCodes = tree.idToCode(sequence)
    val leafIds = ArrayBuffer.empty[(Int, Float)]
    val childNodes = ArrayBuffer.empty[TreeNode]
    val rootNode = TreeNodePred(0, tree.kvData(0.toString), 0.0f)
    var candidateNodes = ArrayBuffer(rootNode)

    while (candidateNodes.nonEmpty) {
      if (candidateNodes.length > candidateNum) {
        candidateNodes = candidateNodes.sortBy(-_.pred)
      }

      var i, num = 0
      while (i < candidateNodes.length && num < candidateNum) {
        val internalNode = candidateNodes(i)
        val children = tree.getChildNodes(internalNode.code)
        if (children.isEmpty) {
          val node = Node.parseFrom(internalNode.rawNode)
          leafIds += Tuple2(node.id, internalNode.pred)
        } else {
          childNodes ++= children
          num += 1
        }
        i += 1
      }

      if (childNodes.isEmpty) {
        candidateNodes.clear()
      } else {
        val feature = transform(seqCodes, childNodes)
        val preds = model.forward(feature).asInstanceOf[Tensor[Float]].storage().array()
        val nextLen = childNodes.length
        candidateNodes.clear()

        i = 0
        while (i < nextLen) {
          candidateNodes += TreeNodePred(childNodes(i).code, childNodes(i).node, preds(i))
          i += 1
        }
        childNodes.clear()
      }
    }

    // leafIds.sortBy(-_._2).take(topk).map(i => (i._1, sigmoid(i._2)))
  }
}
