package com.mass.tdm.operator

import com.mass.tdm.dataset.{SampledMiniBatch, TDMSample}
import com.mass.tdm.tree.DistTree
import com.mass.tdm.utils.NegativeSampler

class TDMOp(val seqLen: Int, layerNegCounts: String, concat: Boolean = true) extends Serializable {
  import TDMOp.{tree, sampler}

  val sampledNodesNumPerTarget: Int = computeSampleUnit()
  lazy val maxCode: Int = tree.maxCode
  @transient private var data: Array[TDMSample] = _

  private def computeSampleUnit(): Int = {
    val _layerNegCounts = layerNegCounts.split(",")
    require(_layerNegCounts.length >= tree.maxLevel, "Not enough negative sample layers")
    require(_layerNegCounts.zipWithIndex.forall { case (num, i) =>
      num.toInt < math.pow(2, i).toInt
    }, "Num of negative samples must not exceed max numbers in current layer")

    val negNumPerLayer = _layerNegCounts.slice(0, tree.maxLevel).map(_.toDouble.toInt)
    // positive(one per layer) + negative nums
    negNumPerLayer.length + negNumPerLayer.sum
  }

  def setData(data: Array[TDMSample]): Unit = {
    this.data = data
  }

  def apply(data: Array[TDMSample], offset: Int, length: Int,
      threadId: Int = 0): (Array[Int], Array[Float]) = {
    val targetItems = (offset until offset + length).map(i => data(i).target).toArray
    val (itemCodes, labels) = sampleNegative(targetItems, threadId)
    val features = transform(data, offset, length, itemCodes)
    (features, labels)
  }

  def sampleNegative(targetItemIds: Array[Int], threadId: Int): (Array[Int], Array[Float]) = {
    val (itemIds, labels) = sampler.sample(targetItemIds, threadId)
    val itemCodes = tree.idToCode(itemIds)
    (itemCodes, labels)
  }

  def transform(data: Array[TDMSample], offset: Int, length: Int,
      targetItems: Array[Int]): Array[Int] = {
    // or fill with maxCode
    val features = Array.fill[Int](length * sampledNodesNumPerTarget * seqLen)(0)
    var dataOffset = offset
    var i = 0
    while (i < length) {
      val featItems = tree.idToCode(data(dataOffset).sequence)
      val offset1 = i * (sampledNodesNumPerTarget * seqLen)
      var j = 0
      while (j < sampledNodesNumPerTarget) {
        val offset2 = offset1 + j * seqLen
        var s = 0
        while (s < seqLen - 1) {
          features(offset2 + s) = featItems(s)
          s += 1
        }
        // seqLen = seq items + target item
        features(offset2 + s) = targetItems(i * sampledNodesNumPerTarget + j)
        j += 1
      }
      i += 1
      dataOffset += 1
    }
    features
  }
}

object TDMOp {

  private var _pbFilePath: String = ""
  private var _layerNegCounts: String = ""
  private var _withProb: Boolean = _
  private var _startSampleLayer: Int = _
  private var _tolerance: Int = _
  private var _numThreads: Int = _
  private var _parallelSample: Boolean = _

  lazy val tree: DistTree = buildTree(_pbFilePath)
  lazy val sampler: NegativeSampler = buildSampler(tree)

  private def init(
      pbFilePath: String,
      layerNegCounts: String,
      withProb: Boolean = true,
      startSampleLayer: Int = -1,
      tolerance: Int = 20,
      numThreads: Int,
      parallelSample: Boolean): Unit = {

    _pbFilePath = pbFilePath
    _layerNegCounts = layerNegCounts
    _withProb = withProb
    _startSampleLayer = startSampleLayer
    _tolerance = tolerance
    _numThreads = numThreads
    _parallelSample = parallelSample
  }

  private def buildTree(pbFilePath: String): DistTree = {
    val singletonTree = DistTree(pbFilePath)
    singletonTree
  }

  private def buildSampler(tree: DistTree): NegativeSampler = {
    val singletonSampler = new NegativeSampler(tree, _layerNegCounts, _withProb,
      _startSampleLayer, _tolerance, _numThreads)
    if (_parallelSample) {
      singletonSampler.initParallel()
    } else {
      singletonSampler.init()
    }
    singletonSampler
  }

  def apply(
      pbFilePath: String,
      layerNegCounts: String,
      withProb: Boolean = true,
      startSampleLayer: Int = -1,
      tolerance: Int = 20,
      numThreads: Int,
      seqLen: Int,
      parallelSample: Boolean,
      concat: Boolean = true): TDMOp = {

    init(pbFilePath, layerNegCounts, withProb, startSampleLayer, tolerance,
      numThreads, parallelSample)
    require(tree.initialized && sampler.initialized, "not properly initialized yet...")
    new TDMOp(seqLen, layerNegCounts, concat)
  }

  def partialApply(pbFilePath: String)(
      layerNegCounts: String,
      withProb: Boolean = true,
      startSampleLayer: Int = -1,
      tolerance: Int = 20,
      numThreads: Int,
      seqLen: Int,
      parallelSample: Boolean,
      concat: Boolean = true): TDMOp = {

    init(pbFilePath, layerNegCounts, withProb, startSampleLayer, tolerance,
      numThreads, parallelSample)
    require(tree.initialized && sampler.initialized, "not properly initialized yet...")
    new TDMOp(seqLen, layerNegCounts, concat)
  }
}
