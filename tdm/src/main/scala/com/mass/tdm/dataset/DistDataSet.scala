package com.mass.tdm.dataset

import java.nio.file.{Files, Paths}
import java.util.concurrent.atomic.AtomicInteger

import scala.collection.mutable

import com.mass.sparkdl.dataset.DataUtil
import com.mass.sparkdl.utils.{Engine, FileReader => DistFileReader}
import com.mass.tdm.dataset.TDMSample.{TDMEvalSample, TDMTrainSample}
import com.mass.tdm.encoding
import com.mass.tdm.operator.TDMOp
import org.apache.log4j.Logger
import org.apache.spark.{SparkContext, SparkFiles}
import org.apache.spark.rdd.RDD

class DistDataSet(
    totalBatchSize: Int,
    totalEvalBatchSize: Int = -1,
    evaluate: Boolean,
    seqLen: Int,
    layerNegCounts: String,
    withProb: Boolean = true,
    startSampleLevel: Int = 1,
    tolerance: Int = 20,
    numThreadsPerNode: Int,
    parallelSample: Boolean,
    partitionNum: Option[Int] = None,
    miniBatch: Option[RDD[MiniBatch]] = None,
    delimiter: String = ",",
    useMask: Boolean = true) {

  val logger: Logger = Logger.getLogger(getClass)

  private val batchSizePerNode = DataUtil.getBatchSize(totalBatchSize, partitionNum)
  private val evalBatchSizePerNode = DataUtil.getBatchSize(totalEvalBatchSize, partitionNum)
  private var dataBuffer: RDD[Array[TDMSample]] = _
  private var evalDataBuffer: RDD[Array[TDMSample]] = _
  private var miniBatchBuffer: RDD[MiniBatch] = miniBatch.orNull
  private var evalMiniBatchBuffer: RDD[MiniBatch] = _
  private[tdm] var isCached: Boolean = false
  private[tdm] var isCachedEval: Boolean = false
  private[tdm] val parallelSampling = parallelSample
  private[tdm] val evaluateDuringTraining = evaluate
  private[tdm] var userConsumed: RDD[Map[Int, Seq[Int]]] = _

  lazy val count: Int = dataBuffer.mapPartitions(iter => {
    require(iter.hasNext)
    val array = iter.next()
    require(!iter.hasNext)
    Iterator.single(array.length)
  }).reduce(_ + _)

  lazy val evalCount: Int = {
    if (evaluateDuringTraining) {
      require(evalDataBuffer != null)
      evalDataBuffer.mapPartitions(iter => {
        val array = iter.next()
        Iterator.single(array.length)
      }).reduce(_ + _)
    } else {
      -1
    }
  }

  lazy val maxCode: Int = miniBatchBuffer.first.maxCode

  def readRDD(
      sc: SparkContext,
      dataPath: String,
      pbFilePath: String,
      evalPath: Option[String] = None,
      userConsumedPath: Option[String] = None): Unit = {

  //  require(Files.exists(Paths.get(dataPath)), s"$dataPath doesn't exist")
  //  require(Files.exists(Paths.get(pbFilePath)), s"$pbFilePath doesn't exist")

    val _partitionNum = partitionNum.getOrElse(Engine.nodeNumber())
    val _delimiter = delimiter
    // todos: read into local, then sc.parallelize
    dataBuffer = sc.textFile(dataPath, _partitionNum)
      .map(_.trim.split(_delimiter))
      .filter(line => (1 until line.length - 1).exists(line(_).toDouble != 0))
      .map(line => {
        val seqItems = line.slice(1, line.length - 1).map(_.toInt)
        val target = line.last.toInt
        TDMTrainSample(seqItems, target).asInstanceOf[TDMSample]
      })
      .coalesce(_partitionNum, shuffle = true)
      .mapPartitions(iter => Iterator.single(iter.toArray))
      .setName("cached train dataset")
      .cache()

    // distribute file to every node
    sc.addFile(pbFilePath)
    if (null == miniBatchBuffer) {
      val _batchSize = batchSizePerNode
      val _seqLen = seqLen
      val _layerNegCounts = layerNegCounts
      val _startSampleLevel = startSampleLevel
      val _useMask = useMask
      val localParams = (layerNegCounts, withProb, startSampleLevel,
        tolerance, numThreadsPerNode)
      // initialize tree on driver
      val pbPath = SparkFiles.get(pbFilePath.split("/").last)
      logger.info("pbFilePath: " + pbPath)
      TDMOp.initTree(pbPath)

      miniBatchBuffer = dataBuffer.mapPartitions(dataIter => {
        val localData = dataIter.next()
        val pbPath = SparkFiles.get(pbFilePath.split("/").last)
        // use partial function with tuple parameters to initialize TDMOp object
        TDMOp.partialApply(pbPath) _ tupled localParams
        // val localTDM = TDMOp.apply _ tupled localParams
        val localMiniBatch = new MiniBatch(_batchSize, _seqLen,
          localData.length, _layerNegCounts, _startSampleLevel, _useMask)
        Iterator.single(localMiniBatch)
      }).setName("miniBatchBuffer").cache()
    }

    if (evaluateDuringTraining) {
      require(evalBatchSizePerNode > 0, "must set evalBatchSize for evaluating")
      evalPath match {
        case Some(path: String) => readEvalRDD(sc, path)
        case _ => throw new IllegalArgumentException("invalid eval data path...")
      }
      userConsumedPath match {
        case Some(path: String) => readUserConsumed(sc, path)
        case _ => throw new IllegalArgumentException("invalid user consumed path...")
      }
    }
  }

  def readEvalRDD(sc: SparkContext, evalPath: String): Unit = {
    val _partitionNum = partitionNum.getOrElse(Engine.nodeNumber())
    val _delimiter = delimiter
    val _seqLen = seqLen + 1
    evalDataBuffer = sc.textFile(evalPath, _partitionNum)
      .map(data => {
        val line = data.trim.split(_delimiter)
        val user = line.head.substring(5).toInt
        val seqItems = line.slice(1, _seqLen).map(_.toInt)
        val labels = line.slice(_seqLen, line.length).map(_.toInt)
        TDMEvalSample(seqItems, labels, user).asInstanceOf[TDMSample]
      })
      .coalesce(_partitionNum, shuffle = true)
      .mapPartitions(iter => Iterator.single(iter.toArray))
      .setName("cached eval dataset")
      .cache()

    if (null == evalMiniBatchBuffer) {
      val _batchSize = evalBatchSizePerNode
      val _seqLen = seqLen
      val _layerNegCounts = layerNegCounts
      val _startSampleLevel = startSampleLevel
      val _useMask = useMask
      evalMiniBatchBuffer = evalDataBuffer.mapPartitions(dataIter => {
        val localData = dataIter.next()
        val localMiniBatch = new MiniBatch(_batchSize, _seqLen,
          localData.length, _layerNegCounts, _startSampleLevel, _useMask)
        Iterator.single(localMiniBatch)
      }).setName("eval miniBatchBuffer").cache()
    }
  }

  def readUserConsumed(sc: SparkContext, userConsumedPath: String): Unit = {
    // val userConsumedLocal = mutable.HashMap.empty[Int, Array[Int]]
    val fileReader = DistFileReader(userConsumedPath)
    val input = fileReader.open()
    val fileInput = scala.io.Source.fromInputStream(input, encoding.name())
    val mapping = for {
      line <- fileInput.getLines
      arr = line.trim.split(delimiter)
      user = arr.head.substring(5).toInt
      items = arr.tail.map(_.toInt)
    } yield user -> items.toSeq
    val userConsumedLocal = mapping.toMap

    fileInput.close()
    input.close()
    fileReader.close()

    val broadcastUserConsumed = sc.broadcast(userConsumedLocal)
    userConsumed = evalDataBuffer.mapPartitions { _ =>
      Iterator.single(broadcastUserConsumed.value)
    }.setName("userConsumed").cache()
  }

  def size(): Int = count

  def evalSize(): Int = evalCount

  def shuffle(): Unit = {
    dataBuffer.unpersist()
    dataBuffer = dataBuffer.mapPartitions(iter => {
      Iterator.single(DataUtil.shuffle(iter.next()))
    }).setName("shuffled data").cache()
    // dataBuffer.count()
  }

  def originalRDD(): RDD[Array[TDMSample]] = dataBuffer

  def originalEvalRDD(): RDD[Array[TDMSample]] = evalDataBuffer

  def userConsumedRDD(): RDD[Map[Int, Seq[Int]]] = userConsumed

  def getMaxCode: Int = maxCode

  def iteratorMiniBatch(train: Boolean, expandBatch: Boolean): RDD[MiniBatch] = {
    if (!train) {
      require(evalDataBuffer != null, "can't evaluate without eval data...")
    }

    val _train = train
    val _batchSize = if (train) batchSizePerNode else evalBatchSizePerNode
    val _miniBatchRDD = if (train) miniBatchBuffer else evalMiniBatchBuffer
    _miniBatchRDD.mapPartitions(miniBatchIter => {
      val miniBatch = miniBatchIter.next()
      // val numTargetsPerBatch = miniBatch.numTargetsPerBatch
      val numTargetsPerBatch = if (expandBatch) miniBatch.numTargetsPerBatch else _batchSize
      val localDataSize = miniBatch.originalDataSize

      new Iterator[MiniBatch] {
        private val index = new AtomicInteger(0)

        override def hasNext: Boolean = {
          if (_train) true else index.get() < localDataSize
        }

        override def next(): MiniBatch = {
          val curIndex = index.getAndAdd(numTargetsPerBatch)
          if (_train || curIndex < localDataSize) {
            val offset = if (_train) curIndex % localDataSize else curIndex
            val length = math.min(numTargetsPerBatch, localDataSize - offset)
            miniBatch.updatePosition(offset, length)
          } else {
            null
          }
        }
      }
    })
  }

  def cache(): Unit = {
    // use action to trigger computation
    dataBuffer.count()
    miniBatchBuffer.count()
    if (null != evalDataBuffer) {
      evalDataBuffer.count()
    }
    if (null != evalMiniBatchBuffer) {
      evalMiniBatchBuffer.count()
    }
    if (null != userConsumed) {
      userConsumed.count()
    }
    isCached = true
  }

  def unpersist(): Unit = {
    dataBuffer.unpersist()
    miniBatchBuffer.unpersist()
    if (null != evalDataBuffer) {
      evalDataBuffer.unpersist()
    }
    if (null != evalMiniBatchBuffer) {
      evalMiniBatchBuffer.unpersist()
    }
    if (null != userConsumed) {
      userConsumed.unpersist()
    }
    isCached = false
  }
}
