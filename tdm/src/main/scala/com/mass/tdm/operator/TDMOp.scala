package com.mass.tdm.operator

import scala.collection.mutable.ArrayBuffer

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
      parallelSample: Boolean): Unit = {

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
      parallelSample: Boolean): Unit = {

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
      seqLen: Int,
      concat: Boolean): (Seq[Array[Int]], Array[Float]) = {

    val targetItems = (offset until offset + length).map(i => data(i).target).toArray
    val (itemCodes, labels) = sampleNegative(targetItems, threadId)
    if (concat) {
      val itemSeqs = transform(data, offset, length, itemCodes, sampledNodesNumPerTarget, seqLen)
      (Seq(itemSeqs), labels)
    } else {
      val (itemSeqs, masks) = transform(data, offset, length, sampledNodesNumPerTarget, seqLen)
      (Seq(itemCodes, itemSeqs, masks), labels)
    }
  }

  def sampleNegative(targetItemIds: Array[Int], threadId: Int)
      : (Array[Int], Array[Float]) = {
    val (itemIds, labels) = sampler.sample(targetItemIds, threadId)
    val itemCodes = tree.idToCode(itemIds)
    (itemCodes, labels)
  }

  def transform(data: Array[TDMSample], offset: Int, length: Int, targetItems: Array[Int],
      sampledNodesNumPerTarget: Int, seqLen: Int): Array[Int] = {

    val features = new Array[Int](length * sampledNodesNumPerTarget * seqLen)
    var dataOffset = offset
    var i = 0
    while (i < length) {
      val featItems = tree.idToCode(data(dataOffset).sequence)
      val offset1 = i * (sampledNodesNumPerTarget * seqLen)
      var j = 0
      while (j < sampledNodesNumPerTarget) {
        val offset2 = offset1 + j * seqLen
        val length = seqLen - 1
        System.arraycopy(featItems, 0, features, offset2, length)
        // seqLen = seq items + target item
        features(offset2 + length) = targetItems(i * sampledNodesNumPerTarget + j)
        j += 1
      }
      i += 1
      dataOffset += 1
    }
    features
  }

  def transform(data: Array[TDMSample], offset: Int, length: Int,
      sampledNodesNumPerTarget: Int, seqLen: Int): (Array[Int], Array[Int]) = {

    val features = new Array[Int](length * sampledNodesNumPerTarget * seqLen)
    val maskBuffer = new ArrayBuffer[Int]()
    var dataOffset = offset
    var i = 0
    while (i < length) {
      val (featItems, mask) = tree.idToCodeWithMask(data(dataOffset).sequence)
      val offset1 = i * (sampledNodesNumPerTarget * seqLen)
      var j = 0
      while (j < sampledNodesNumPerTarget) {
        val offset2 = offset1 + j * seqLen
        System.arraycopy(featItems, 0, features, offset2, seqLen)
        mask.foreach(m => maskBuffer += (m + offset2))
        j += 1
      }
      i += 1
      dataOffset += 1
    }
    (features, maskBuffer.toArray)
  }
}
