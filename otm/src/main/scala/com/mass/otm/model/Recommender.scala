package com.mass.otm.model

import com.mass.otm.dataset.OTMSample
import com.mass.otm.dataset.MiniBatch.{duplicateSequence, transform}
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
    useMask: Boolean,
  ): Seq[Seq[Node]] = {
    val (initCandidates, initSize) = tree.initializeBeam(batchData.length)
    val (itemSeqs, masks) = duplicateSequence(batchData, beamSize, seqLen)
    (tree.startLevel until tree.leafLevel).foldLeft(initCandidates) { case (candidateNodes, level) =>
      val (batchInputs, beamNodes, candidateNum) = {
        if (level == tree.startLevel) {
          val nodes = candidateNodes.map(_.flatMap(i => Seq(i.id * 2 + 1, i.id * 2 + 2)))
          val inputs = transform(batchData, nodes, initSize, None, seqLen, useMask)
          (inputs, nodes, initSize * 2)
        } else {
          val nodes =
            for {
              nodes <- candidateNodes
            } yield {
              nodes
                .sortBy(_.pred)(Ordering[Double].reverse)
                .take(beamSize)
                .flatMap(n => Seq(n.id * 2 + 1, n.id * 2 + 2))
            }
          val inputs = transform(batchData, nodes, beamSize, Some((itemSeqs, masks)), seqLen, useMask)
          (inputs, nodes, beamSize * 2)
        }
      }
      val batchOutputs = deepModel.forward(batchInputs).toTensor
      val candidatePreds = batchOutputs
        .storage()
        .array()
        .sliding(candidateNum, candidateNum)
        .toSeq

      beamNodes.zip(candidatePreds).map { case (nodes, preds) =>
        nodes.lazyZip(preds).map(Node)
      }
    }
  }
}
