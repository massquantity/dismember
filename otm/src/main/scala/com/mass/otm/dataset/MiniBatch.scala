package com.mass.otm.dataset

import com.mass.otm.paddingIdx
import com.mass.otm.tree.OTMTree.TargetNode
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.utils.Table

object MiniBatch {

  def cumSum[A](xs: Seq[A])(implicit num: Numeric[A]): Seq[A] = {
    xs.tail.scanLeft(xs.head)(num.plus)
  }

  // batchSize * beamSize * 2 * seqLen
  def duplicateSequence(data: Seq[OTMSample], nodeSize: Int, seqLen: Int): (Tensor[Int], Tensor[Int]) = {
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

  def transform(
    data: Seq[OTMSample],
    nodes: Seq[Seq[Int]],
    targets: Seq[TargetNode],
    seqLen: Int,
    useMask: Boolean
  )(
    itemSeqs: Option[(Tensor[Int], Tensor[Int])],
    nodeSize: Int
  ): (Table, Tensor[Double], Int) = {
    val shape = Array(data.length * nodeSize * 2, 1)
    val labels =
      for {
        (levelNodes, target) <- nodes zip targets
        n <- levelNodes
        label = if (n == target.id) target.label else 0.0
      } yield label

    val (batchItemSeqs, masks) = itemSeqs match {
      case Some(i) => i
      case None => duplicateSequence(data, nodeSize, seqLen)
    }
    val batchItems = Tensor(nodes.flatten.toArray, shape)
    val batchLabels = Tensor(labels.toArray, shape)
    if (useMask) {
      (Table(batchItems, batchItemSeqs, masks), batchLabels, nodeSize * 2)
    } else {
      (Table(batchItems, batchItemSeqs), batchLabels, nodeSize * 2)
    }
  }

  def transform(
    data: Seq[OTMSample],
    nodes: Seq[Seq[Int]],
    nodeSize: Int,
    itemSeqs: Option[(Tensor[Int], Tensor[Int])],
    seqLen: Int,
    useMask: Boolean
  ): Table = {
    val shape = Array(data.length * nodeSize * 2, 1)
    val (batchItemSeqs, masks) = itemSeqs match {
      case Some(i) => i
      case None => duplicateSequence(data, nodeSize, seqLen)
    }
    val batchItems = Tensor(nodes.flatten.toArray, shape)
    if (useMask) {
      Table(batchItems, batchItemSeqs, masks)
    } else {
      Table(batchItems, batchItemSeqs)
    }
  }
}
