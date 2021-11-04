package com.mass.tdm.dataset

import java.nio.file.{Files, Paths}
import java.util.concurrent.atomic.AtomicInteger

import scala.collection.mutable

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
    useMask: Boolean = true) {

  private val batchSize = totalBatchSize
  private val evalBatchSize = totalEvalBatchSize
  private var dataBuffer: Array[TDMSample] = _
  private var evalDataBuffer: Array[TDMSample] = _
  private var miniBatchBuffer: MiniBatch = miniBatch.orNull
  private var evalMiniBatchBuffer: MiniBatch = _
  private[tdm] val parallelSampling = parallelSample
  private[tdm] val evaluateDuringTraining = evaluate
  private[tdm] val userConsumed = new mutable.HashMap[Int, Array[Int]]()

  def readFile(
      dataPath: String,
      pbFilePath: String,
      evalPath: Option[String] = None,
      userConsumedPath: Option[String] = None): Unit = {

    val fileReader = DistFileReader(dataPath)
    val input = fileReader.open()
    val fileInput = scala.io.Source.fromInputStream(input, encoding.name())
    dataBuffer = (
      for {
        line <- fileInput.getLines
        arr = line.trim.split(delimiter)
        seqItems = arr.slice(1, arr.length - 1)
        target = arr.last
        if seqItems.exists(_.toDouble.toInt != 0)
      } yield TDMTrainSample(seqItems.map(_.toInt), target.toInt)
    ).toArray

    fileInput.close()
    input.close()
    fileReader.close()

    if (null == miniBatchBuffer) {
      TDMOp(pbFilePath, layerNegCounts, withProb, startSampleLayer,
        tolerance, numThreads, parallelSample)
      miniBatchBuffer = new MiniBatch(batchSize, seqLen,
        dataBuffer.length, layerNegCounts, useMask)
    }

    if (evaluateDuringTraining) {
      require(evalBatchSize > 0, "must set evalBatchSize for evaluating")
      evalPath match {
        case Some(path: String) => readEvalFile(path)
        case _ => throw new IllegalArgumentException("invalid evaluate data path...")
      }
      userConsumedPath match {
        case Some(path: String) => readUserConsumed(path)
        case _ => throw new IllegalArgumentException("invalid user consumed path...")
      }
    }
  }

  def readEvalFile(evalPath: String): Unit = {
    // item sequence range [1, seqLen + 1), sequence + target
    val _seqLen = seqLen + 1
    val fileReader = DistFileReader(evalPath)
    val input = fileReader.open()
    val fileInput = scala.io.Source.fromInputStream(input, encoding.name())
    evalDataBuffer = (
      for {
        line <- fileInput.getLines
        arr = line.trim.split(delimiter)
        user = arr.head.substring(5).toInt
        seqItems = arr.slice(1, _seqLen).map(_.toInt)
        labels = arr.slice(_seqLen, arr.length).map(_.toInt)
      } yield TDMEvalSample(seqItems, labels, user)
    ).toArray

    fileInput.close()
    input.close()
    fileReader.close()

    if (null == evalMiniBatchBuffer ) {
      evalMiniBatchBuffer = new MiniBatch(evalBatchSize, seqLen,
        evalDataBuffer.length, layerNegCounts, useMask)
    }
  }

  def readUserConsumed(userConsumedPath: String): Unit = {
    val fileReader = DistFileReader(userConsumedPath)
    val input = fileReader.open()
    val fileInput = scala.io.Source.fromInputStream(input, encoding.name())
    for {
      line <- fileInput.getLines
      arr = line.trim.split(delimiter)
      user = arr.head.substring(5).toInt
      items = arr.tail.map(_.toInt)
    } {
      userConsumed(user) = items
    }

    fileInput.close()
    input.close()
    fileReader.close()
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

  def getUserConsumed: mutable.HashMap[Int, Array[Int]] = userConsumed
}

object LocalDataSet {
  def apply(): LocalDataSet = {
    ???
  }
}
