package com.mass.tdm.dataset

import com.mass.sparkdl.tensor.Tensor
import com.mass.tdm.operator.TDMOp

class MiniBatch(
  //  @transient val tdmConverter: TDMOp,
    batchSize: Int,
    val seqLen: Int,
    val originalDataSize: Int,
    layerNegCounts: String) extends Serializable {

  @transient private var features: Array[Tensor[Int]] = _
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
      threadId: Int): (Array[Tensor[Int]], Tensor[Float]) = {
    val (features, labels) = TDMOp.convert(data, threadOffset, threadLen,
      threadId, sampledNodesNumPerTarget, seqLen)
    val featureShape = Array(threadLen * sampledNodesNumPerTarget, seqLen)
    val labelShape = Array(threadLen * sampledNodesNumPerTarget)
    (Array(Tensor(features, featureShape)), Tensor(labels, labelShape))
  }

  def convertAll(data: Array[TDMSample]): this.type = {
    val (_features, _labels) = TDMOp.convert(data, offset, length, 0,
      sampledNodesNumPerTarget, seqLen)
    val featureShape = Array(length * sampledNodesNumPerTarget, seqLen)
    val labelShape = Array(length * sampledNodesNumPerTarget)
    this.features = Array(Tensor(_features, featureShape))
    this.labels = Tensor(_labels, labelShape)
    this
  }

  def slice(offset: Int, length: Int): (Array[Tensor[Int]], Tensor[Float]) = {
    (Array(features.head.narrow(0, offset, length)), labels.narrow(0, offset, length))
  }

  def getOffset: Int = offset

  def getLength: Int = length

  def getFeatures: Array[Tensor[Int]] = features

  def getLabels: Tensor[Float] = labels

}
