package com.mass.otm.tree

import com.mass.otm.{paddingIdx, DeepModel}
import com.mass.otm.dataset.OTMSample
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble
import com.mass.sparkdl.utils.Table

// start level has num of nodes <= beamSize, i.e. upper level of first candidate level
class OTMTree(val startLevel: Int, val leafLevel: Int) {
  import OTMTree._

  // batchSize * initSize
  def initializeBeam(batchSize: Int): (Seq[Seq[Node]], Int) = {
    val startNode = (1 to startLevel).foldLeft(0)((i, _) => i * 2 + 1)
    val endNode = startNode * 2 + 1
    val initNodes = Seq.range(startNode, endNode).map(Node(_, 0.0))
    val initCandidates = Seq.fill(batchSize)(initNodes)
    val initSize = endNode - startNode
    (initCandidates, initSize)
  }

  // levelSize * batchSize
  def optimalPseudoTargets(data: Seq[OTMSample], seqLen: Int, useMask: Boolean)(
    implicit model: DeepModel[Double]
  ): List[List[TargetNode]] = {
    val (itemSeqs, masks) = sequenceBatch(data, seqLen)
    (leafLevel until startLevel by -1).foldLeft[List[List[TargetNode]]](Nil) { (allNodes, _) =>
      val levelNodes = allNodes match {
        case Nil => data.map(i => TargetNode(i.target, 1.0)).toList
        case childNodes :: _ => computeTargets(childNodes, itemSeqs, masks, useMask)
      }
      levelNodes :: allNodes
    }
  }
}

object OTMTree {

  case class Node(id: Int, pred: Double)

  case class TargetNode(id: Int, label: Double)

  def apply(startLevel: Int, leafLevel: Int): OTMTree = {
    new OTMTree(startLevel, leafLevel)
  }

  def computeTargets(
    childNodes: List[TargetNode],
    itemSeqs: Tensor[Int],
    masks: Tensor[Int],
    useMask: Boolean
  )(
    implicit model: DeepModel[Double]
  ): List[TargetNode] = {
    val (posPreds, negPreds) = computeChildScores(childNodes, itemSeqs, masks, useMask)
    childNodes.lazyZip(posPreds).lazyZip(negPreds).map { case (n, pos, neg) =>
      val parentId = (n.id - 1) >> 1
      val parentLabel = if (pos > neg) n.label else 0.0
      TargetNode(parentId, parentLabel)
    }
  }

  def computeChildScores(
    nodes: List[TargetNode],
    itemSeqs: Tensor[Int],
    masks: Tensor[Int],
    useMask: Boolean
  )(
    implicit model: DeepModel[Double]
  ): (Array[Double], Array[Double]) = {
    val length = nodes.length
    val posNodes = nodes.map(_.id).toArray
    val negNodes = posNodes.map(n => if (n % 2 == 0) n - 1 else n + 1)
    val posTensor = Tensor(posNodes, Array(length, 1))
    val negTensor = Tensor(negNodes, Array(length, 1))
    val (posInputs, negInputs) =
      if (useMask) {
        (Table(posTensor, itemSeqs, masks), Table(negTensor, itemSeqs, masks))
      } else {
        (Table(negTensor, itemSeqs), Table(negTensor, itemSeqs))
      }
    val posPreds = model.forward(posInputs).toTensor.storage().array()
    val negPreds = model.forward(negInputs).toTensor.storage().array()
    (posPreds, negPreds)
  }

  def sequenceBatch(data: Seq[OTMSample], seqLen: Int): (Tensor[Int], Tensor[Int]) = {
    val sequence = data.flatMap(_.sequence).toArray
    val masks = sequence.zipWithIndex.filter(_._1 == paddingIdx).map(_._2)
    val maskTensor = if (masks.isEmpty) Tensor[Int]() else Tensor(masks, Array(masks.length))
    (Tensor(sequence, Array(data.length, seqLen)), maskTensor)
  }
}
