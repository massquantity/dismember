package com.mass.sparkdl.utils

import java.io.InputStream
import java.util.Locale

import com.intel.analytics.bigdl.mkl.MKL
import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, SparkContext, SparkException}

object Engine {

  private val logger = Logger.getLogger(getClass)

  @volatile private var _default: ThreadPool = _

  @volatile private var _model: ThreadPool = new ThreadPool(1)

  private var physicalCoreNum: Int = -1
  private var nodeNum: Int = 1

  def coreNumber(): Int = {
    require(physicalCoreNum != -1, s"Engine.init: Core number is not initialized")
    physicalCoreNum
  }

  private[sparkdl] def setCoreNumber(n: Int): Unit = {
    require(n > 0, "Engine.init: core number is smaller than zero")
    physicalCoreNum = n
    initThreadPool(n)
  }

  def nodeNumber(): Int = {
    require(nodeNum != -1, s"Engine.init: Node number is not initialized")
    nodeNum
  }

  private[sparkdl] def setNodeNumber(n : Int): Unit = {
    require(n > 0)
    nodeNum = n
  }

  def model: ThreadPool = {
    _model
  }

  def default: ThreadPool = {
    if (_default == null) {
      throw new IllegalStateException(s"Engine.init: Thread engine is not initialized.")
    }
    _default
  }

  def setNodeAndCore(nodeNum: Int, coreNum: Int): Unit = {
    setNodeNumber(nodeNum)
    setCoreNumber(coreNum)
  }

  private def initThreadPool(core: Int): Unit = {
    val defaultPoolSize: Int = System.getProperty("sparkdl.defaultPoolSize", core.toString).toInt
    if (_default == null || _default.getPoolSize != defaultPoolSize) {
      _default = new ThreadPool(defaultPoolSize)
    }

    val modelPoolSize = 1
    if(_model == null || _model.getPoolSize != modelPoolSize) {
      _model = new ThreadPool(modelPoolSize)
    }
    _model.setMKLThread(MKL.getMklNumThreads)
  }

}
