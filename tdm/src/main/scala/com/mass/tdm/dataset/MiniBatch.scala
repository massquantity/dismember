package com.mass.tdm.dataset

import com.mass.sparkdl.nn.abstractnn.Activity
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.utils.T
import com.mass.tdm.operator.TDMOp

class MiniBatch(
  //  @transient val tdmConverter: TDMOp,
    batchSize: Int,
    val seqLen: Int,
    val originalDataSize: Int,
    layerNegCounts: String,
    concat: Boolean) extends Serializable {

  @transient private var features: Tensor[Int] = _
  // @transient private var featuresGroup: Seq[Tensor[Int]] = _
  @transient private var labels: Tensor[Float] = _
  private var offset: Int = -1
  private var length: Int = -1
  val sampledNodesNumPerTarget: Int = computeSampleUnit()
  val numTargetsPerBatch: Int = math.max(1, batchSize / sampledNodesNumPerTarget)
  lazy val maxCode: Int = TDMOp.tree.maxCode

  private def computeSampleUnit(): Int = {
    val _layerNegCounts = layerNegCounts.split(",")
    require(_layerNegCounts.length >= TDMOp.tree.maxLevel, "Not enough negative sample layers")
    require(_layerNegCounts.zipWithIndex.forall { case (num, i) =>
      num.toInt < math.pow(2, i).toInt
    }, "Num of negative samples must not exceed max numbers in current layer")

    val negNumPerLayer = _layerNegCounts.slice(0, TDMOp.tree.maxLevel).map(_.toDouble.toInt)
    // positive(one per layer) + negative nums
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
      threadId: Int): (Activity, Tensor[Float]) = {

    val (features, labels) = TDMOp.convert(data, threadOffset, threadLen,
      threadId, sampledNodesNumPerTarget, seqLen, concat)
    val itemSeqShape = Array(threadLen * sampledNodesNumPerTarget, seqLen)
    val labelShape = Array(threadLen * sampledNodesNumPerTarget)

    if (concat) {
      val itemSeqs = features.head
      (Tensor(itemSeqs, itemSeqShape), Tensor(labels, labelShape))
    } else {
      val Seq(targetItems, itemSeqs, masks) = features
      val itemShape = Array(threadLen * sampledNodesNumPerTarget, 1)
      val maskShape = Array(masks.length)
      val convertedTable = T(Tensor(targetItems, itemShape),
        Tensor(itemSeqs, itemSeqShape), Tensor(masks, maskShape))
      (convertedTable, Tensor(labels, labelShape))
    }
  }

  def convertAll(data: Array[TDMSample]): this.type = {
    require(concat, "Attention mode only support parallel sampling")
    val (_features, _labels) = TDMOp.convert(data, offset, length, 0,
      sampledNodesNumPerTarget, seqLen, concat)
    val itemSeqShape = Array(length * sampledNodesNumPerTarget, seqLen)
    val labelShape = Array(length * sampledNodesNumPerTarget)
    this.features = Tensor(_features.head, itemSeqShape)
    this.labels = Tensor(_labels, labelShape)
    this
  }

  def slice(offset: Int, length: Int): (Tensor[Int], Tensor[Float]) = {
    (features.narrow(0, offset, length), labels.narrow(0, offset, length))
  }

  def getOffset: Int = offset

  def getLength: Int = length

  def getFeatures: Tensor[Int] = features

  def getLabels: Tensor[Float] = labels

}
