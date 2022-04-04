package com.mass.tdm.dataset

import com.mass.sparkdl.tensor.Tensor
import com.mass.tdm.operator.TDMOp

class MiniBatch(
    batchSize: Int,
    val seqLen: Int,
    val originalDataSize: Int,
    layerNegCounts: String,
    startSampleLevel: Int,
    useMask: Boolean) extends Serializable {
  import MiniBatch._

  private var offset: Int = -1
  private var length: Int = -1
  val sampledNodesNumPerTarget: Int = computeSampleUnit()
  val numTargetsPerBatch: Int = math.max(1, batchSize / sampledNodesNumPerTarget)
  lazy val maxCode: Int = TDMOp.tree.maxCode

  private def computeSampleUnit(): Int = {
    val _layerNegCounts = layerNegCounts.split(",")
    require(_layerNegCounts.length >= TDMOp.tree.maxLevel + 1, "Not enough negative sample layers")
    require(_layerNegCounts.zipWithIndex.forall { case (num, i) =>
      num.toInt < math.pow(2, i).toInt
    }, "Num of negative samples must not exceed max numbers in current layer")

    // positive(one per layer) + negative nums, exclude root node
    val negNumPerLayer = _layerNegCounts.slice(startSampleLevel, TDMOp.tree.maxLevel + 1).map(_.toInt)
    negNumPerLayer.length + negNumPerLayer.sum
  }

  def expandedSize(): Int = {
    require(length != -1, "length is not valid, please update position first...")
    length * sampledNodesNumPerTarget
  }

  def updatePosition(_offset: Int, _length: Int): this.type = {
    this.offset = _offset
    this.length = _length
    this
  }

  def convert(
    data: Array[TDMSample],
    threadOffset: Int,
    threadLen: Int,
    threadId: Int,
  ): TransformedBatch = {
    val targetItems = Seq.range(threadOffset, threadOffset + threadLen).map(data(_).target)
    val (itemCodes, labels) = sampleNegative(targetItems, threadId)
    val (itemSeqs, seqMasks) =
      if (!useMask) {
        transform(data, threadOffset, threadLen, sampledNodesNumPerTarget)
      } else {
        transformWithMask(data, threadOffset, threadLen, sampledNodesNumPerTarget, seqLen)
      }
    val itemShape = Array(threadLen * sampledNodesNumPerTarget, 1)
    val itemSeqShape = Array(threadLen * sampledNodesNumPerTarget, seqLen)
    val labelShape = Array(threadLen * sampledNodesNumPerTarget)

    if (!useMask) {
      SeqTransformedBatch(
        Tensor(itemCodes.toArray, itemShape),
        Tensor(itemSeqs.toArray, itemSeqShape),
        Tensor(labels.toArray, labelShape),
      )
    } else {
      val masks =
        if (seqMasks.isEmpty) {
          Tensor[Int]()
        } else {
          val maskShape = Array(seqMasks.length)
          Tensor(seqMasks, maskShape)
        }
      MaskTransformedBatch(
        Tensor(itemCodes.toArray, itemShape),
        Tensor(itemSeqs.toArray, itemSeqShape),
        Tensor(labels.toArray, labelShape),
        masks,
      )
    }
  }

  def getOffset: Int = offset

  def getLength: Int = length

}

object MiniBatch {

  sealed trait TransformedBatch extends Product with Serializable

  case class SeqTransformedBatch(
    items: Tensor[Int],
    sequence: Tensor[Int],
    labels: Tensor[Float]) extends TransformedBatch

  case class MaskTransformedBatch(
    items: Tensor[Int],
    sequence: Tensor[Int],
    labels: Tensor[Float],
    masks: Tensor[Int]) extends TransformedBatch

  def sampleNegative(targetItemIds: Seq[Int], threadId: Int): (Seq[Int], Seq[Float]) = {
    val (itemCodes, labels) = TDMOp.sampler.sample(targetItemIds, threadId)
    // val (itemCodes, _) = TDMOp.tree.idToCode(itemIds)
    (itemCodes, labels)
  }

  def transform(
    data: Array[TDMSample],
    offset: Int,
    length: Int,
    sampledNodesNumPerTarget: Int,
  ): (Seq[Int], Array[Int]) = {
    val copiedSeqs = Seq.range(offset, offset + length) flatMap { i =>
      val (featItems, _) = TDMOp.tree.idToCode(data(i).sequence)
      Seq.fill(sampledNodesNumPerTarget)(featItems).flatten
    }
    (copiedSeqs, Array.empty[Int])
  }

  def transformWithMask(
    data: Array[TDMSample],
    offset: Int,
    length: Int,
    sampledNodesNumPerTarget: Int,
    seqLen: Int,
  ): (Seq[Int], Array[Int]) = {
    val (copiedSeqs, masks) = (
      for {
        i <- 0 until length
        itemIds = data(offset + i).sequence
        (featItems, mask) = TDMOp.tree.idToCode(itemIds)
        dataOffset = i * (sampledNodesNumPerTarget * seqLen)
        j <- 0 until sampledNodesNumPerTarget
        offsetMask = mask.map(_ + dataOffset + j * seqLen)
      } yield (featItems, offsetMask)
    ).unzip
    (copiedSeqs.flatten, masks.toArray.flatten)
  }
}
