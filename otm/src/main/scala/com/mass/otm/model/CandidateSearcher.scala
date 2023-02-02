package com.mass.otm.model

import com.mass.otm.{lowerLog2, paddingIdx, DeepModel}
import com.mass.otm.dataset.{MiniBatch, OTMSample}
import com.mass.otm.tree.OTMTree
import com.mass.otm.tree.OTMTree.Node
import com.mass.scalann.tensor.Tensor
import com.mass.scalann.tensor.TensorNumeric.NumericDouble
import com.mass.scalann.utils.Table

trait CandidateSearcher {
  import CandidateSearcher._

  // batchSize * beamSize * 2
  private[otm] def batchBeamSearch(
      batchData: Seq[OTMSample],
      deepModel: DeepModel[Double],
      tree: OTMTree,
      beamSize: Int,
      seqLen: Int,
      useMask: Boolean
  ): List[List[Node]] = {
    val (initNodes, initSize) = tree.initializeBeam(batchData.length)
    val miniBatch = MiniBatch(batchData, beamSize, tree.startLevel)(seqLen, useMask)
    (tree.startLevel until tree.leafLevel).foldLeft(initNodes) { case (beamNodes, level) =>
      val candidateNodes =
        if (level == tree.startLevel) {
          beamNodes.map(_.flatMap(n => List(n.id * 2 + 1, n.id * 2 + 2)))
        } else {
          for {
            nodes <- beamNodes
          } yield {
            nodes
              .sortBy(_.score)(Ordering[Double].reverse)
              .take(beamSize)
              .flatMap(n => List(n.id * 2 + 1, n.id * 2 + 2))
          }
        }
      val nodeSize = if (level == tree.startLevel) initSize else beamSize
      val batchInputs = miniBatch.batchTransform(candidateNodes, level == tree.startLevel)
      val batchOutputs = deepModel.forward(batchInputs).toTensor
      // take certain size since the underlying array may be larger than u think
      val offset = batchOutputs.storageOffset()
      val end = offset + batchOutputs.nElement()
      val candidatePreds = batchOutputs
        .storage()
        .array()
        .slice(offset, end)
        .sliding(nodeSize * 2, nodeSize * 2)
        .toSeq

      candidateNodes.zip(candidatePreds).map { case (nodes, preds) =>
        nodes.lazyZip(preds).map(Node)
      }
    }
  }

  private[otm] def beamSearch(
      sequence: Seq[Int],
      deepModel: DeepModel[Double],
      leafLevel: Int,
      beamSize: Int,
      useMask: Boolean
  ): Seq[Node] = {
    val startLevel = lowerLog2(beamSize)
    val startNode = (1 to startLevel).foldLeft(0)((i, _) => i * 2 + 1)
    val endNode = startNode * 2 + 1
    val initNodes = Seq.range(startNode, endNode).map(Node(_, 0.0))
    (startLevel until leafLevel).foldLeft(initNodes) { case (beamNodes, level) =>
      val (inputs, candidateNodes) = buildInputs(
        sequence,
        beamNodes,
        beamSize,
        level == startLevel,
        useMask
      )
      val candidatePreds = deepModel.forward(inputs).toTensor.storage().array()
      candidateNodes.lazyZip(candidatePreds).map(Node)
    }
  }
}

object CandidateSearcher {

  def buildInputs(
      sequence: Seq[Int],
      nodes: Seq[Node],
      beamSize: Int,
      beamStart: Boolean,
      useMask: Boolean
  ): (Table, Seq[Int]) = {
    val candidateNodes = buildBeamNodes(nodes, beamSize, beamStart)
    val nodesTensor = Tensor(candidateNodes.toArray, Array(candidateNodes.length, 1))
    val itemSeqs = Array.range(0, candidateNodes.length).flatMap(_ => sequence)
    val seqTensor = Tensor(itemSeqs, Array(candidateNodes.length, sequence.length))
    if (useMask) {
      val masks = itemSeqs.zipWithIndex.filter(_._1 == paddingIdx).map(_._2)
      val maskTensor = if (masks.isEmpty) {
        Tensor[Int]()
      } else {
        Tensor(masks, Array(masks.length))
      }
      (Table(nodesTensor, seqTensor, maskTensor), candidateNodes)
    } else {
      (Table(nodesTensor, seqTensor), candidateNodes)
    }
  }

  def buildBeamNodes(
      nodes: Seq[Node],
      beamSize: Int,
      beamStart: Boolean
  ): Seq[Int] = {
    if (beamStart) {
      nodes.flatMap(n => Seq(n.id * 2 + 1, n.id * 2 + 2))
    } else {
      nodes
        .sortBy(_.score)(Ordering[Double].reverse)
        .take(beamSize)
        .flatMap(n => Seq(n.id * 2 + 1, n.id * 2 + 2))
    }
  }

}
