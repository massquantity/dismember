package com.mass.tdm.dataset

import java.util.concurrent.atomic.AtomicInteger

import com.mass.scalann.utils.{FileReader => DistFileReader}
import com.mass.scalann.utils.DataUtil
import com.mass.tdm.dataset.TDMSample.{TDMTrainSample, TDMEvalSample}
import com.mass.tdm.encoding
import com.mass.tdm.operator.TDMOp

class LocalDataSet(
    trainPath: String,
    evalPath: String,
    pbFilePath: String,
    userConsumedPath: String,
    totalTrainBatchSize: Int,
    totalEvalBatchSize: Int,
    seqLen: Int,
    layerNegCounts: String,
    withProb: Boolean = true,
    startSampleLevel: Int = 1,
    tolerance: Int = 20,
    numThreads: Int = 1,
    useMask: Boolean = true) {
  import LocalDataSet._
  require(startSampleLevel > 0, s"start sample level should be at least 1, got $startSampleLevel")

  private val trainBatchSize = totalTrainBatchSize
  private val evalBatchSize = totalEvalBatchSize
  private val trainData: Array[TDMSample] = readFile(trainPath, seqLen, readTrainData)
  private val evalData: Array[TDMSample] = readFile(evalPath, seqLen, readEvalData)
  val userConsumed: Map[Int, Seq[Int]] = readFile(userConsumedPath, seqLen, readUserConsumed)
  lazy val trainMiniBatch: MiniBatch = new MiniBatch(
    trainBatchSize,
    seqLen,
    trainData.length,
    layerNegCounts,
    startSampleLevel,
    useMask
  )
  lazy val evalMiniBatch: MiniBatch = new MiniBatch(
    evalBatchSize,
    seqLen,
    evalData.length,
    layerNegCounts,
    startSampleLevel,
    useMask
  )
  // initialize TDMOp
  TDMOp(
    pbFilePath,
    layerNegCounts,
    withProb,
    startSampleLevel,
    tolerance,
    numThreads
  )

  def iteratorMiniBatch(train: Boolean, expandBatch: Boolean): Iterator[MiniBatch] = {
    if (!train) {
      require(evalData != null, "can't evaluate without eval data...")
    }

    new Iterator[MiniBatch] {
      private val _miniBatch = if (train) trainMiniBatch else evalMiniBatch
      private val _batchSize =  if (train) trainBatchSize else evalBatchSize
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

  def getMaxCode: Int = trainMiniBatch.maxCode

  def shuffle(): Unit = DataUtil.shuffle(trainData)

  def trainSize: Int = trainData.length

  def evalSize: Int = evalData.length

  def getData: Array[TDMSample] = trainData

  def getEvalData: Array[TDMSample] = evalData

  def getUserConsumed: Map[Int, Seq[Int]] = userConsumed
}

object LocalDataSet {

  def apply(
      trainPath: String,
      evalPath: String,
      pbFilePath: String,
      userConsumedPath: String,
      totalTrainBatchSize: Int,
      totalEvalBatchSize: Int,
      seqLen: Int,
      layerNegCounts: String,
      withProb: Boolean = true,
      startSampleLevel: Int = 1,
      tolerance: Int = 20,
      numThreads: Int = 1,
      useMask: Boolean = true
  ): LocalDataSet = {
    new LocalDataSet(
      trainPath,
      evalPath,
      pbFilePath,
      userConsumedPath,
      totalTrainBatchSize,
      totalEvalBatchSize,
      seqLen,
      layerNegCounts,
      withProb,
      startSampleLevel,
      tolerance,
      numThreads,
      useMask)
  }

  def readFile[T](path: String, seqLen: Int, f: (scala.io.Source, Int) => T): T = {
    val fileReader = DistFileReader(path)
    val input = fileReader.open()
    val fileInput = scala.io.Source.fromInputStream(input, encoding.name())
    val output = f(fileInput, seqLen)
    fileInput.close()
    input.close()
    fileReader.close()
    output
  }

  val readTrainData: (scala.io.Source, Int) => Array[TDMSample] = (source, _) => {
    val samples =
      for {
        line <- source.getLines()
        arr = line.trim.split(",")
        seqItems = arr.slice(1, arr.length - 1)
        target = arr.last
        if seqItems.exists(_.toDouble.toInt != 0)
      } yield TDMTrainSample(seqItems.map(_.toInt), target.toInt)
    samples.toArray
  }

  val readEvalData: (scala.io.Source, Int) => Array[TDMSample] = (source, seqLen) => {
    // item sequence range [1, seqLen + 1), sequence + target
    val samples =
      for {
        line <- source.getLines()
        arr = line.trim.split(",")
        user = arr.head.substring(5).toInt
        seqItems = arr.slice(1, seqLen + 1).map(_.toInt)
        labels = arr.slice(seqLen + 1, arr.length).map(_.toInt)
      } yield TDMEvalSample(seqItems, labels, user)
    samples.toArray
  }

  val readUserConsumed: (scala.io.Source, Int) => Map[Int, Seq[Int]] = (source, _) => {
    val mapping =
      for {
        line <- source.getLines()
        arr = line.trim.split(",")
        user = arr.head.substring(5).toInt
        items = arr.tail.map(_.toInt).toSeq
      } yield user -> items
    mapping.toMap
  }
}
