package com.mass.otm.dataset

import com.mass.otm.paddingIdx
import com.mass.otm.tree.OTMTree.TargetNode
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.utils.Table

class MiniBatch(
    batchData: IndexedSeq[OTMSample],
    initSize: Int,
    beamSize: Int,
    seqLen: Int,
    useMask: Boolean,
    numThread: Int) {
  import MiniBatch._
  val threadDataSize = math.ceil(batchData.length.toDouble / numThread).toInt
  val (initItemSeqs, initMasks) = batchSequence(initSize)
  val (itemSeqs, masks) = batchSequence(beamSize)

  def batchSequence(nodeSize: Int): (IndexedSeq[Tensor[Int]], IndexedSeq[Tensor[Int]]) = {
    batchData
      .sliding(threadDataSize, threadDataSize)
      .map(duplicateSequence(_, nodeSize, seqLen))
      .toIndexedSeq
      .unzip
  }

  def batchTransform(
    nodes: Seq[Seq[Int]],
    targets: Seq[TargetNode],
    nodeSize: Int
  ): IndexedSeq[(Table, Tensor[Double])] = {
    val batchItemSeqs = if (nodeSize == initSize) initItemSeqs else itemSeqs
    val batchMasks = if (nodeSize == initSize) initMasks else masks
    val batchNum = threadDataSize * nodeSize * 2
    val batchItems = nodes
      .flatten
      .toArray
      .sliding(batchNum, batchNum)
      .map(items => Tensor(items, Array(items.length, 1)))
      .toIndexedSeq
    val allLabels =
      for {
        (levelNodes, target) <- nodes zip targets
        n <- levelNodes
        label = if (n == target.id) target.label else 0.0
      } yield label
    val batchLabels = allLabels
      .toArray
      .sliding(batchNum, batchNum)
      .map(labels => Tensor(labels, Array(labels.length, 1)))
      .toIndexedSeq

    (0 until numThread).map { i =>
      if (useMask) {
        (Table(batchItems(i), batchItemSeqs(i), batchMasks(i)), batchLabels(i))
      } else {
        (Table(batchItems(i), batchItemSeqs(i)), batchLabels(i))
      }
    }
  }

  def batchTransform(nodes: Seq[Seq[Int]], nodeSize: Int): Table = {
    val shape = Array(nodes.map(_.length).sum, 1)
    val batchItems = Tensor(nodes.flatten.toArray, shape)
    val batchItemSeqs = if (nodeSize == initSize) initItemSeqs.head else itemSeqs.head
    val batchMasks = if (nodeSize == initSize) initMasks.head else masks.head
    if (useMask) {
      Table(batchItems, batchItemSeqs, batchMasks)
    } else {
      Table(batchItems, batchItemSeqs)
    }
  }
}

object MiniBatch {

  def apply(
    batchData: IndexedSeq[OTMSample],
    initSize: Int,
    beamSize: Int,
    seqLen: Int,
    useMask: Boolean,
    numThread: Int
  ): MiniBatch = {
    new MiniBatch(
      batchData,
      initSize,
      beamSize,
      seqLen,
      useMask,
      numThread
    )
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
}
