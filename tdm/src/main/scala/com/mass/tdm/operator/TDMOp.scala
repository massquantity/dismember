package com.mass.tdm.operator

import com.mass.tdm.tree.TDMTree
import com.mass.tdm.utils.NegativeSampler
import org.apache.log4j.Logger

object TDMOp {

  val logger: Logger = Logger.getLogger(getClass)

  private var _pbFilePath: String = ""
  private var _layerNegCounts: String = ""
  private var _withProb: Boolean = _
  private var _startSampleLevel: Int = _
  private var _tolerance: Int = _
  private var _numThreads: Int = _

  lazy val tree: TDMTree = buildTree(_pbFilePath)
  lazy val sampler: NegativeSampler = buildSampler(tree)

  def initTree(pbFilePath: String): Unit = {
    _pbFilePath = pbFilePath
    require(tree.initialized)
  }

  private def init(
    pbFilePath: String,
    layerNegCounts: String,
    withProb: Boolean = true,
    startSampleLevel: Int = 1,
    tolerance: Int = 20,
    numThreads: Int
  ): Unit = {
    _pbFilePath = pbFilePath
    _layerNegCounts = layerNegCounts
    _withProb = withProb
    _startSampleLevel = startSampleLevel
    _tolerance = tolerance
    _numThreads = numThreads
  }

  private def buildTree(pbFilePath: String): TDMTree = {
    require(pbFilePath != "", "must initialize tree first...")
    val singletonTree = TDMTree(pbFilePath)
    singletonTree
  }

  private def buildSampler(tree: TDMTree): NegativeSampler = {
    val singletonSampler = new NegativeSampler(
      tree,
      _layerNegCounts,
      _withProb,
      _startSampleLevel,
      _tolerance,
      _numThreads
    )
    singletonSampler.initParallel()
  }

  def apply(
    pbFilePath: String = "",
    layerNegCounts: String = "",
    withProb: Boolean = true,
    startSampleLevel: Int = 1,
    tolerance: Int = 20,
    numThreads: Int = 1
  ): Unit = {
    init(
      pbFilePath,
      layerNegCounts,
      withProb,
      startSampleLevel,
      tolerance,
      numThreads
    )
    require(tree.initialized && sampler.initialized, "not properly initialized yet...")
    logger.info(s"item num: ${tree.getIds.length}, tree level: ${tree.getMaxLevel}, " +
      s"generated node number for one target: ${sampler.layerSum}")
  }

  def partialApply(pbFilePath: String)(
    layerNegCounts: String,
    withProb: Boolean = true,
    startSampleLevel: Int = 1,
    tolerance: Int = 20,
    numThreads: Int
  ): Unit = {
    init(
      pbFilePath,
      layerNegCounts,
      withProb,
      startSampleLevel,
      tolerance,
      numThreads
    )
    require(tree.initialized && sampler.initialized, "not properly initialized yet...")
    logger.info(s"item num: ${tree.getIds.length}, tree level: ${tree.getMaxLevel}, " +
      s"generated node number for one target: ${sampler.layerSum}")
  }
}
