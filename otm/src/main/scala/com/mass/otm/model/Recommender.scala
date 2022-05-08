package com.mass.otm.model

import com.mass.otm.dataset.{MiniBatch, OTMSample}
import com.mass.otm.DeepModel
import com.mass.otm.tree.OTMTree
import com.mass.otm.tree.OTMTree.Node
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble

trait Recommender {

  // batchSize * beamSize * 2
  private[otm] def batchBeamSearch(
    batchData: IndexedSeq[OTMSample],
    deepModel: DeepModel[Double],
    tree: OTMTree,
    beamSize: Int,
    seqLen: Int,
    useMask: Boolean
  ): Seq[Seq[Node]] = {
    val (initCandidates, initSize) = tree.initializeBeam(batchData.length)
    val miniBatch = MiniBatch(batchData, initSize, beamSize, seqLen, useMask, 1)
    (tree.startLevel until tree.leafLevel).foldLeft(initCandidates) { case (candidateNodes, level) =>
      val beamNodes =
        if (level == tree.startLevel) {
          candidateNodes.map(_.flatMap(i => Seq(i.id * 2 + 1, i.id * 2 + 2)))
        } else {
          for {
            nodes <- candidateNodes
          } yield {
            nodes
              .sortBy(_.pred)(Ordering[Double].reverse)
              .take(beamSize)
              .flatMap(n => Seq(n.id * 2 + 1, n.id * 2 + 2))
          }
        }
      val nodeSize = if (level == tree.startLevel) initSize else beamSize
      val batchInputs = miniBatch.batchTransform(beamNodes, nodeSize)
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

      beamNodes.zip(candidatePreds).map { case (nodes, preds) =>
        nodes.lazyZip(preds).map(Node)
      }
    }
  }
}
