package com.mass.scalann.utils

import com.intel.analytics.bigdl.mkl.MKL

object Engine {

  @volatile private var _default: ThreadPool = _

  @volatile private var _model: ThreadPool = _

  private var physicalCoreNum: Int = -1

  def coreNumber(): Int = {
    require(physicalCoreNum != -1, s"Engine.init: Core number is not initialized")
    physicalCoreNum
  }

  def setCoreNumber(n: Int): Unit = {
    require(n > 0, "Engine.init: core number is smaller than zero")
    physicalCoreNum = n
    initThreadPool(n)
    System.setProperty("scala.concurrent.context.numThreads", s"$n")
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

  private def initThreadPool(core: Int): Unit = {
    if (_default == null || _default.getPoolSize != core) {
      _default = new ThreadPool(core)
    }

    val modelPoolSize = 1
    if(_model == null || _model.getPoolSize != modelPoolSize) {
      _model = new ThreadPool(modelPoolSize)
    }
    _model.setMKLThread(MKL.getMklNumThreads)
  }
}
