package com.mass.tdm.operator

import com.mass.tdm.dataset.TDMSample
import com.mass.tdm.tree.DistTree
import com.mass.tdm.utils.NegativeSampler
import org.apache.log4j.Logger

object TDMOp {

  val logger: Logger = Logger.getLogger(getClass)

  private var _pbFilePath: String = ""
  private var _layerNegCounts: String = ""
  private var _withProb: Boolean = _
  private var _startSampleLayer: Int = _
  private var _tolerance: Int = _
  private var _numThreads: Int = _
  private var _parallelSample: Boolean = _

  lazy val tree: DistTree = buildTree(_pbFilePath)
  lazy val sampler: NegativeSampler = buildSampler(tree)

  def initTree(pbFilePath: String): Unit = {
    _pbFilePath = pbFilePath
    require(tree.initialized)
  }

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
    require(pbFilePath != "", "must initialize tree first...")
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
      pbFilePath: String = "",
      layerNegCounts: String = "",
      withProb: Boolean = true,
      startSampleLayer: Int = -1,
      tolerance: Int = 20,
      numThreads: Int = 1,
      parallelSample: Boolean,
      concat: Boolean = true): Unit = {

    init(pbFilePath, layerNegCounts, withProb, startSampleLayer, tolerance,
      numThreads, parallelSample)
    require(tree.initialized && sampler.initialized, "not properly initialized yet...")
    logger.info(s"item num: ${tree.getIds.length}, tree level: ${tree.getMaxLevel}, " +
      s"generated node number for one target: ${sampler.layerSum}")
  }

  def partialApply(pbFilePath: String)(
      layerNegCounts: String,
      withProb: Boolean = true,
      startSampleLayer: Int = -1,
      tolerance: Int = 20,
      numThreads: Int,
      parallelSample: Boolean,
      concat: Boolean = true): Unit = {

    init(pbFilePath, layerNegCounts, withProb, startSampleLayer, tolerance,
      numThreads, parallelSample)
    require(tree.initialized && sampler.initialized, "not properly initialized yet...")
    logger.info(s"item num: ${tree.getIds.length}, tree level: ${tree.getMaxLevel}, " +
      s"generated node number for one target: ${sampler.layerSum}")
  }

  def convert(
      data: Array[TDMSample],
      offset: Int,
      length: Int,
      threadId: Int,
      sampledNodesNumPerTarget: Int,
      seqLen: Int): (Array[Int], Array[Float]) = {
    val targetItems = (offset until offset + length).map(i => data(i).target).toArray
    val (itemCodes, labels) = sampleNegative(targetItems, threadId)
    val features = transform(data, offset, length, itemCodes, sampledNodesNumPerTarget, seqLen)
    (features, labels)
  }

  def sampleNegative(targetItemIds: Array[Int], threadId: Int): (Array[Int], Array[Float]) = {
    val (itemIds, labels) = sampler.sample(targetItemIds, threadId)
    val itemCodes = tree.idToCode(itemIds)
    (itemCodes, labels)
  }

  def transform(data: Array[TDMSample], offset: Int, length: Int, targetItems: Array[Int],
      sampledNodesNumPerTarget: Int, seqLen: Int): Array[Int] = {
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
