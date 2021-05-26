package com.mass.tdm.optim

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future

import com.mass.sparkdl.{Criterion, Module}
import com.mass.sparkdl.optim.{OptimMethod, Trigger}
import com.mass.sparkdl.parameters.AllReduceParameter
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.utils.{CachedModels, Engine, ModelBroadcast, T, Table}
import com.mass.tdm.dataset.{DistDataSet, MiniBatch, TDMSample}
import com.mass.tdm.evaluation.Evaluator
import com.mass.tdm.optim.OptimUtil._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.TaskContext

class DistOptimizer(
    model: Module[Float],
    dataset: DistDataSet,
    criterion: Criterion[Float],
    optimMethod: OptimMethod[Float],
    numIteration: Int,
    progressInterval: Int,
    topk: Int,
    candidateNum: Int,
    concat: Boolean) {

  private val state: Table = T()
  private val endWhen: Trigger = Trigger.maxIteration(numIteration, "trainIter")
  private var cachedModelsRDD: RDD[Cache[Float]] = _
  private var modelBroadcast: ModelBroadcast[Float] = _
  private var reserveOptimMethod: Boolean = false
  private var previousOptim: RDD[OptimMethod[Float]] = _

  def reserveOptim(reserve: Boolean): this.type = {
    reserveOptimMethod = reserve
    this
  }

  private def resetOptimMethod(models: RDD[Cache[Float]],
      prevOptimMethods: RDD[OptimMethod[Float]]): RDD[Cache[Float]] = {

    models.zipPartitions(prevOptimMethods)((modelIter, optimIter) => {
      val cache = modelIter.next()
      cache.optimMethod = optimIter.next()
      Iterator.single(cache)
    })
  }

  def optimize(): Module[Float] = {
    optimMethod.clearHistory()
    val nodeNumber = Engine.nodeNumber()
    val coresPerNode = Engine.coreNumber()
    val partitionNum = dataset.originalRDD().getNumPartitions
    val (modelWeights, _): (Tensor[Float], Tensor[Float]) = model.adjustParameters()
    val allReduceParameter = AllReduceParameter.newParameter[Float](
      partitionNum, modelWeights.nElement(), compress = "fp16")

    if (!dataset.isCached) {
      dataset.cache()  // cache both train and eval data
    }

    val (initializedModels, broadcastModels) = DistOptimizer.initThreadModels(
      model, dataset, criterion, nodeNumber, coresPerNode, allReduceParameter, optimMethod)

    cachedModelsRDD = {
      if (reserveOptimMethod && previousOptim != null) {
        resetOptimMethod(initializedModels, previousOptim)
      } else {
        initializedModels
      }
    }
    modelBroadcast = broadcastModels

    DistOptimizer.optimizeImpl(model, dataset, coresPerNode, endWhen, cachedModelsRDD,
      optimMethod, allReduceParameter, progressInterval, topk, candidateNum, concat)

    DistOptimizer.getModel(cachedModelsRDD, allReduceParameter, model)

    shutdown()

    if (reserveOptimMethod) {
      previousOptim = cachedModelsRDD.map(m => m.optimMethod).cache()
      previousOptim.count()
    } else {
      if (previousOptim != null) {
        previousOptim.unpersist()
      }
    }
    cachedModelsRDD.unpersist()
    model
  }

  private def shutdown(): Unit = {
    cachedModelsRDD.foreachPartition { iter =>
      iter.foreach(_.localModels.foreach(_.release()))
    }
    CachedModels.deleteKey[Float](modelBroadcast.uuid)
  }
}

object DistOptimizer {
  val logger: Logger = Logger.getLogger(getClass)
  // logger.setLevel(Level.INFO)

  private def initThreadModels(
      model: Module[Float],
      dataset: DistDataSet,
      criterion: Criterion[Float],
      nodeNumber: Int,
      coresPerNode: Int,
      allReduceParameter: AllReduceParameter[Float],
      optimMethod: OptimMethod[Float]): (RDD[Cache[Float]], ModelBroadcast[Float]) = {
    require(dataset.originalRDD().getNumPartitions == nodeNumber,
      s"RDD partition number ${dataset.originalRDD().getNumPartitions}" +
        s" is not equal to configured node number $nodeNumber")

    val sc = dataset.originalRDD().sparkContext
    val broadcast = sc.broadcast((criterion, optimMethod))
    model.adjustParameters()
    val modelBroadcast = ModelBroadcast[Float]().broadcast(sc, model)

    // local variables for RDD operation
    val subModelNum = coresPerNode
    val nExecutor = Engine.nodeNumber()
    val executorCores = Engine.coreNumber()

    val models = dataset.originalRDD().mapPartitions(_ => {
      val partitionId = TaskContext.getPartitionId()
      val (broadcastCriterion, broadcastOptim) = broadcast.value
      Engine.setNodeAndCore(nExecutor, executorCores)
      // one model per core
      val cached = (0 until subModelNum).map(_ => {
        val localModel = modelBroadcast.value(initGradient = true, shareWeight = true)
        localModel.setId(partitionId)
        val localCriterion = broadcastCriterion.cloneCriterion()
        val (weights, grads) = localModel.adjustParameters()
        (localModel, weights, grads, localCriterion)
      }).toArray

      // localModels are same, weights are shared
      val weights = cached.head._2
      allReduceParameter.init(weights)

      Iterator.single(Cache(
        cached.map(_._1), // models in each node/partition, one model per cpu core
        cached.map(_._2), // weights
        cached.map(_._3), // gradients
        cached.map(_._4), // criterions
        broadcastOptim.clone()
      ))
    }).persist()

    models.setName("Thread Model RDD")
    logger.info("Cache thread models...")
    // to trigger persist
    models.count()
    logger.info("Cache thread models... done")
    (models, modelBroadcast)
  }

  private def optimizeImpl(
      originalModel: Module[Float], // original defined model to train
      dataset: DistDataSet,
      coresPerNode: Int,
      endWhen: Trigger,
      models: RDD[Cache[Float]], // cached models in each node
      optimMethod: OptimMethod[Float],
      parameters: AllReduceParameter[Float],
      progressInterval: Int,
      topk: Int,
      candidateNum: Int,
      concat: Boolean): Unit = {

    val sc = dataset.originalRDD().sparkContext
    val partitionNum = dataset.originalRDD().getNumPartitions
    val subModelNum = coresPerNode
    var epochTime = 0L

    val driverState = initState(optimMethod, subModelNum)
    logger.info("Count dataset")
    val countBefore = System.nanoTime()
    val numSamples = dataset.size()
    logger.info(s"numSamples: $numSamples")
    val countAfter = System.nanoTime()
    logger.info(f"Count dataset complete. Time elapsed: ${(countAfter - countBefore) / 1e9d}%.4fs")

    var recordsProcessedThisEpoch = optimMethod.state[Long]("recordsProcessedThisEpoch")
    if (recordsProcessedThisEpoch == 0) {
      val shuffleBefore = System.nanoTime()
      logger.info("shuffle data")
      dataset.shuffle()
      val shuffleEnd = System.nanoTime()
      logger.info(f"Shuffle data complete. Takes ${(shuffleEnd - shuffleBefore) / 1e9d}%.4fs")
    }

    val tasks: ArrayBuffer[Future[_]] = new ArrayBuffer()
    var miniBatchRDD = dataset.iteratorMiniBatch(train = true, expandBatch = true)
    val dataRDD = dataset.originalRDD()
    val parallelConvert: Boolean = dataset.parallelSampling

    while (!endWhen(driverState)) {
      val lossSum = sc.doubleAccumulator("loss sum")
      val recordsNum = sc.longAccumulator("record number")
      val start = System.nanoTime()

      val numFinishedModelUpdates = miniBatchRDD.zipPartitions(
        dataRDD, models, preservesPartitioning = true) { (miniBatchIter, dataIter, modelIter) =>
          val cachedModel: Cache[Float] = modelIter.next()
          // Copy weights from block manager to local cached model for training,
          // local models per thread share the same weights,
          // so only need to copy weights to the first model.
          // Original weights are split into parts and placed on different partitions/nodes,
          // so these weight parts will be copied to local models separately.
          val weightResults = parameters.getWeights(cachedModel.modelWeights.head)

          val data: Array[TDMSample] = dataIter.next()
          val batch: MiniBatch = miniBatchIter.next()
          val (miniBatchBuffer, miniBatchLen, parallelism) = convertBatch(data, batch,
            parallelConvert, subModelNum, tasks)

          Engine.default.sync(tasks)
          weightResults.waitResult()
          tasks.clear()

          val finishedThreads = trainBatch(miniBatchBuffer, miniBatchLen,
            cachedModel, lossSum, recordsNum, parallelism)

          syncGradients(finishedThreads, cachedModel, parameters, subModelNum)

          (0 until subModelNum).foreach(i => {
            cachedModel.localModels(i).training()
            cachedModel.localModels(i).zeroGradParameters()
          })

          Iterator.single(finishedThreads.length)

      }.reduce(_ + _)

      val lossMean = lossSum.value / numFinishedModelUpdates
      driverState("numFinishedModelUpdates") = numFinishedModelUpdates
      driverState("loss") = lossMean.toFloat

      updateParameters(models, parameters, driverState)

      driverState("trainIter") = driverState[Int]("trainIter") + 1
      val end = System.nanoTime()
      val iterationTime = end - start
      epochTime += iterationTime
      recordsProcessedThisEpoch += recordsNum.value

      if (progressInterval > 0 && driverState[Int]("trainIter") % progressInterval == 0) {
        reportProgress(dataset, models, parameters, topk, candidateNum,
          driverState, recordsProcessedThisEpoch, iterationTime, epochTime, lossMean, concat)
      }

      if (recordsProcessedThisEpoch >= numSamples) {
        reportProgress(dataset, models, parameters, topk, candidateNum,
          driverState, recordsProcessedThisEpoch, iterationTime, epochTime, lossMean, concat)

        driverState("epoch") = driverState[Int]("epoch") + 1
        dataset.shuffle()
        miniBatchRDD = dataset.iteratorMiniBatch(train = true, expandBatch = true)
        epochTime = 0L
        recordsProcessedThisEpoch = 0L
      }

      updateOptimState(driverState, recordsProcessedThisEpoch, optimMethod)
    }
  }

  private def initState(optimMethod: OptimMethod[Float], subModelNum: Int): Table = {
    if (!optimMethod.state.contains("epoch")) {
      optimMethod.state.update("epoch", 0)
    }
    if (!optimMethod.state.contains("trainIter")) {
      optimMethod.state.update("trainIter", 0)
    }
    if (!optimMethod.state.contains("loss")) {
      optimMethod.state.update("loss", Float.PositiveInfinity)
    }
    // if (!optimMethod.state.contains("score")) {
    // optimMethod.state.update("score", 0f)
    // }
    if (!optimMethod.state.contains("recordsProcessedThisEpoch")) {
      optimMethod.state.update("recordsProcessedThisEpoch", 0L)
    }

    T("epoch" -> optimMethod.state("epoch"),
      "trainIter" -> optimMethod.state("trainIter"),
      "loss" -> optimMethod.state("loss"),
      //  "score" -> optimMethods.values.head.state("score"),
      "parallelism" -> subModelNum)
  }

  def getModel(
      cachedModels: RDD[Cache[Float]],
      parameters: AllReduceParameter[Float],
      trainingModel: Module[Float]): Module[Float] = {
    // Util.setExtraParametersFromModelRDD(cachedModels, trainingModel, maxSize = 5e8.toInt)
    val (originalWeights, originalGradients) = trainingModel.parameters()
    for (i <- originalGradients.indices) {
      originalGradients(i).resizeAs(originalWeights(i))
    }
    val (parameter, gradientParameter) = trainingModel.adjustParameters()

    val (weights, gradients) = cachedModels.mapPartitions(_ => {
      val curPartitionId = TaskContext.getPartitionId()
      Iterator.single(
        (Map(curPartitionId -> parameters.weightPartition),
         Map(curPartitionId -> parameters.gradientPartition))
      )
    }).reduce((a, b) => (a._1 ++ b._1, a._2 ++ b._2))

    val partitionNum = cachedModels.getNumPartitions
    val taskSize = parameters.size / partitionNum
    require(taskSize != 0)
    val extraSize = parameters.size % partitionNum
    (0 until partitionNum).map { pid =>
      val start = parameters.paramOffset + pid * taskSize + math.min(pid, extraSize)
      val length = taskSize + (if (pid < extraSize) 1 else 0)
      parameter.narrow(0, start, length).copy(weights(pid))
      gradientParameter.narrow(0, start, length).copy(gradients(pid))
    }
    trainingModel
  }

  private def updateOptimState(
      driverState: Table,
      recordsProcessedThisEpoch: Long,
      optimMethod: OptimMethod[Float]): Unit = {

    optimMethod.state.update("recordsProcessedThisEpoch", recordsProcessedThisEpoch)
    optimMethod.state.update("epoch", driverState[Int]("epoch"))
    optimMethod.state.update("trainIter", driverState[Int]("trainIter"))
    optimMethod.state.update("loss", driverState[Float]("loss"))
  }

  private def reportProgress(
      dataset: DistDataSet,
      models: RDD[Cache[Float]],
      parameters: AllReduceParameter[Float],
      topk: Int,
      candidateNum: Int,
      state: Table,
      dataCount: Long,
      iterationTime: Long,
      epochTime: Long,
      trainLoss: Double,
      concat: Boolean): Unit = {

    val progressInfo = new StringBuilder
    progressInfo ++= f"Epoch ${state[Int]("epoch") + 1} train time: ${epochTime / 1e9d}%.4fs, "
    progressInfo ++= s"count/total: $dataCount/${dataset.size()}, "
    progressInfo ++= f"Iteration ${state[Int]("trainIter")} time: ${iterationTime / 1e9d}%.4fs, "
    progressInfo ++= f"Train loss: $trainLoss%.4f\n"

    if (dataset.evaluateDuringTraining) {
      require(dataset.parallelSampling, "must use parallel sampling in train data when evaluating")
      val evalStart = System.nanoTime()
      val evalResult = Evaluator.evaluateRDD(models, dataset, parameters, topk, candidateNum, state, concat)
      val evalEnd = System.nanoTime()
      progressInfo ++= f"\teval time: ${(evalEnd - evalStart) / 1e9d}%.4fs, " +
        f"Metrics: $evalResult\n"
    }

    logger.info(progressInfo.toString)
  }
}
