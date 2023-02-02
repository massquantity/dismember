package com.mass.otm.dataset

import com.mass.otm.paddingIdx
import com.mass.otm.tree.OTMTree.BatchNodes
import com.mass.scalann.tensor.Tensor
import com.mass.scalann.utils.Table

class MiniBatch(batchData: Seq[OTMSample], beamSize: Int, startLevel: Int)(implicit
    seqLen: Int,
    useMask: Boolean
) {
  import MiniBatch._
  val (initItemSeqs, initMasks) = duplicateSequence(batchData, getInitSize(startLevel), seqLen)
  val (itemSeqs, masks) = duplicateSequence(batchData, beamSize, seqLen)

  def batchTransform(
      batchNodes: BatchNodes, // batchSize * beamSize
      batchTargets: BatchNodes, // batchSize * labelNum
      beamStart: Boolean
  ): (Table, Tensor[Double]) = {
    val batchItemSeqs = if (beamStart) initItemSeqs else itemSeqs
    val batchMasks = if (beamStart) initMasks else masks
    val shape = Array(batchNodes.map(_.length).sum, 1)
    val batchItems = Tensor(batchNodes.flatten.toArray.map(_.id), shape)
    val allLabels =
      for {
        (levelNodes, levelTargets) <- batchNodes zip batchTargets
        n <- levelNodes
      } yield levelTargets.find(_.id == n.id) match {
        case Some(i) => i.score
        case None => 0.0
      }
    val batchLabels = Tensor(allLabels.toArray, Array(allLabels.length, 1))
    if (useMask) {
      (Table(batchItems, batchItemSeqs, batchMasks), batchLabels)
    } else {
      (Table(batchItems, batchItemSeqs), batchLabels)
    }
  }

  def batchTransform(nodes: Seq[Seq[Int]], beamStart: Boolean): Table = {
    val shape = Array(nodes.map(_.length).sum, 1)
    val batchItems = Tensor(nodes.flatten.toArray, shape)
    val batchItemSeqs = if (beamStart) initItemSeqs else itemSeqs
    val batchMasks = if (beamStart) initMasks else masks
    if (useMask) {
      Table(batchItems, batchItemSeqs, batchMasks)
    } else {
      Table(batchItems, batchItemSeqs)
    }
  }
}

object MiniBatch {

  def apply(batchData: Seq[OTMSample], initSize: Int, beamSize: Int)(implicit
      seqLen: Int,
      useMask: Boolean
  ): MiniBatch = {
    new MiniBatch(batchData, initSize, beamSize)
  }

  // batchSize * beamSize * 2 * seqLen
  def duplicateSequence(
      data: Seq[OTMSample],
      nodeSize: Int,
      seqLen: Int
  ): (Tensor[Int], Tensor[Int]) = {
    val shape = Array(data.length * nodeSize * 2, seqLen)
    val itemSeqs =
      for {
        d <- data.toArray
        _ <- 1 to (nodeSize * 2)
        i <- d.sequence
      } yield i
    val masks = itemSeqs.zipWithIndex.filter(_._1 == paddingIdx).map(_._2)
    val maskTensor = if (masks.isEmpty) Tensor[Int]() else Tensor(masks, Array(masks.length))
    (Tensor(itemSeqs, shape), maskTensor)
  }

  def getInitSize(startLevel: Int): Int = {
    val startNode = (1 to startLevel).foldLeft(0)((i, _) => i * 2 + 1)
    val endNode = startNode * 2 + 1
    endNode - startNode
  }
}
