package com.mass.tdm.dataset

import java.nio.file.{Files, Paths}
import java.util.concurrent.atomic.AtomicInteger

import scala.collection.mutable.ArrayBuffer

import com.mass.sparkdl.utils.{FileReader => DistFileReader}
import com.mass.sparkdl.dataset.DataUtil
import com.mass.tdm.encoding
import com.mass.tdm.operator.TDMOp

class LocalDataSet(
    totalBatchSize: Int,
    totalEvalBatchSize: Int = -1,
    evaluate: Boolean,
    seqLen: Int,
    layerNegCounts: String,
    withProb: Boolean = true,
    startSampleLayer: Int = -1,
    tolerance: Int = 20,
    numThreads: Int = 1,
    parallelSample: Boolean = false,
    miniBatch: Option[MiniBatch] = None,
    delimiter: String = ",",
    concat: Boolean = true) {

  private val batchSize = totalBatchSize
  private val evalBatchSize = totalEvalBatchSize
  private var dataBuffer: Array[TDMSample] = _
  private var evalDataBuffer: Array[TDMSample] = _
  private var miniBatchBuffer: MiniBatch = miniBatch.orNull
  private var evalMiniBatchBuffer: MiniBatch = _
  private[tdm] val parallelSampling = parallelSample
  private[tdm] var evaluateDuringTraining = evaluate

  def readFile(dataPath: String, pbFilePath: String, evalPath: Option[String] = None): Unit = {
  //  require(Files.exists(Paths.get(dataPath)), s"$dataPath doesn't exist")
  //  require(Files.exists(Paths.get(pbFilePath)), s"$pbFilePath doesn't exist")

    val buffer = new ArrayBuffer[TDMSample]()
    val fileReader = DistFileReader(dataPath)
    val input = fileReader.open()
    val fileInput = scala.io.Source.fromInputStream(input, encoding.name())
    for {
      line <- fileInput.getLines
      arr = line.trim.split(delimiter)
      seqItems = arr.slice(1, arr.length - 1)
      target = arr.last
      if seqItems.exists(_.toDouble.toInt != 0)
    } {
      val sample = TDMTrainSample(seqItems.map(_.toInt), target.toInt)
      buffer += sample
    }

    fileInput.close()
    input.close()
    fileReader.close()
    dataBuffer = buffer.toArray

    if (null == miniBatchBuffer) {
      TDMOp(pbFilePath, layerNegCounts, withProb, startSampleLayer,
        tolerance, numThreads, parallelSample)
      miniBatchBuffer = new MiniBatch(batchSize, seqLen,
        dataBuffer.length, layerNegCounts, concat)
    }

    if (evaluateDuringTraining) {
      require(evalPath.isDefined)
      require(evalBatchSize > 0, "must set evalBatchSize for evaluating")
      readEvalFile(evalPath.get)
    }
  }

  def readEvalFile(evalPath: String): Unit = {
  //  require(Files.exists(Paths.get(evalPath)), s"$evalPath doesn't exist")
    val _seqLen = if (concat) seqLen else seqLen + 1
    val buffer = new ArrayBuffer[TDMEvalSample]()
    val fileReader = DistFileReader(evalPath)
    val input = fileReader.open()
    val fileInput = scala.io.Source.fromInputStream(input, encoding.name())
    for {
      line <- fileInput.getLines
      arr = line.trim.split(delimiter)
      seqItems = arr.slice(1, _seqLen).map(_.toInt)
      labels = arr.slice(_seqLen, arr.length).map(_.toInt)
    } {
      val sample = TDMEvalSample(seqItems, labels)
      buffer += sample
    }

    fileInput.close()
    input.close()
    fileReader.close()
    evalDataBuffer = buffer.toArray

    if (null == evalMiniBatchBuffer ) {
      evalMiniBatchBuffer = new MiniBatch(evalBatchSize, seqLen,
        evalDataBuffer.length, layerNegCounts, concat)
    }
  }

  def shuffle(): Unit = {
    DataUtil.shuffle(dataBuffer)
  }

  def size(): Int = dataBuffer.length

  def evalSize(): Int = evalDataBuffer.length

  def iteratorMiniBatch(train: Boolean, expandBatch: Boolean): Iterator[MiniBatch] = {
    if (!train) {
      require(evalDataBuffer != null, "can't evaluate without eval data...")
    }

    new Iterator[MiniBatch] {
      private val _miniBatch = if (train) miniBatchBuffer else evalMiniBatchBuffer
      private val _batchSize =  if (train) batchSize else evalBatchSize
      private val numTargetsPerBatch = if (expandBatch) _miniBatch.numTargetsPerBatch else _batchSize
      private val index = new AtomicInteger(0)

      override def hasNext: Boolean = {
        if (train) true else index.get() < _miniBatch.originalDataSize
      }

      override def next(): MiniBatch = {
        val curIndex = index.getAndAdd(numTargetsPerBatch)
        if (train || curIndex < _miniBatch.originalDataSize) {
          val offset = if (train) curIndex % _miniBatch.originalDataSize else curIndex
          val length = math.min(numTargetsPerBatch, _miniBatch.originalDataSize - offset)
          _miniBatch.updatePosition(offset, length)
        } else {
          null
        }
      }
    }
  }

  def getMaxCode: Int = miniBatchBuffer.maxCode

  def getData: Array[TDMSample] = dataBuffer

  def getEvalData: Array[TDMSample] = evalDataBuffer
}

object LocalDataSet {
  def apply(): LocalDataSet = {
    ???
  }
}
