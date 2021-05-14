package com.mass.sparkdl.optim

import java.io.{File, FilenameFilter}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future
import scala.reflect.ClassTag

import com.mass.sparkdl.{Criterion, Module}
import com.mass.sparkdl.dataset.{DistributedDataSet, MiniBatch}
import com.mass.sparkdl.nn.Module
import com.mass.sparkdl.parameters.AllReduceParameter
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}
import com.mass.sparkdl.utils._
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD
import org.apache.spark.TaskContext

object DistriOptimizer extends AbstractOptimizer {
  val logger: Logger = Logger.getLogger(getClass)

  abstract class Cache[T] {
    def localModels: Array[Module[T]]
    def modelWeights: Array[Tensor[T]]
    def modelGradients: Array[Tensor[T]]
    def localCriterions: Array[Criterion[T]]
    // def localMethods: Array[Option[Array[ValidationMethod[T]]]]
    def optimMethods: Map[String, OptimMethod[T]]
    def moduleTimeList: Array[Long]
    def parameterSynchronizer: DistriParameterSynchronizer[T]
  }

  case class CacheV1[T](
    localModels: Array[Module[T]],
    modelWeights: Array[Tensor[T]],
    modelGradients: Array[Tensor[T]],
    localCriterions: Array[Criterion[T]],
    localStates: Array[Table],
    var moduleTimeList: Array[Long] = null,
    // localMethods: Array[Option[Array[ValidationMethod[T]]]],
    var optimMethods: Map[String, OptimMethod[T]],
    parameterSynchronizer: DistriParameterSynchronizer[T] = null
  ) extends Cache[T]

  private[optim] def optimize[T: ClassTag](
      trainingModel: Module[T],
      dataset: DistributedDataSet[MiniBatch[T]],
      coresPerNode: Int,
      state: Table,
      endWhen: Trigger,
      models: RDD[CacheV1[T]],
      optimMethods: Map[String, OptimMethod[T]],
      parameters: AllReduceParameter[T],
      parameterSplits: Map[String, (Int, Int)]
  )(implicit ev: TensorNumeric[T]): Unit = {
    val sc = dataset.originRDD().sparkContext
    val partitionNum = dataset.originRDD().partitions.length
    var wallClockTime = 0L
    var lastEpochTime = 0L

    optimMethods.values.foreach { optimMethod =>
      if (!optimMethod.state.contains("epoch")) {
        optimMethod.state.update("epoch", 1)
      }
      if (!optimMethod.state.contains("neval")) {
        optimMethod.state.update("neval", 1)
      }
      if (!optimMethod.state.contains("Loss")) {
        optimMethod.state.update("Loss", Float.PositiveInfinity)
      }
      // if (!optimMethod.state.contains("score")) {
      // optimMethod.state.update("score", 0f)
      // }
      if (!optimMethod.state.contains("recordsProcessedThisEpoch")) {
        optimMethod.state.update("recordsProcessedThisEpoch", 0)
      }
    }

    val _subModelNumber = coresPerNode
    val driverState = T(
      "epoch" -> optimMethods.values.head.state("epoch"),
      "neval" -> optimMethods.values.head.state("neval"),
      "Loss" -> optimMethods.values.head.state("Loss"),
    //  "score" -> optimMethods.values.head.state("score"),
      "parallelism" -> _subModelNumber
    )

    logger.info("Count dataset")
    val countBefore = System.nanoTime()
    val numSamples = dataset.data(train = false).map(_.size()).reduce(_ + _)
    val countAfter = System.nanoTime()
    logger.info(s"Count dataset complete. Time elapsed: ${(countAfter - countBefore) / 1e9}s")

    logger.info(s"config $state")
    var recordsProcessedThisEpoch = optimMethods.values.head.state[Int]("recordsProcessedThisEpoch")
    if (recordsProcessedThisEpoch == 0) {
      val shuffleBefore = System.nanoTime()
      logger.info("Shuffle data")
      dataset.shuffle()
      val shuffleEnd = System.nanoTime()
      logger.info(s"Shuffle data complete. Takes ${(shuffleEnd - shuffleBefore) / 1e9}s")
    }

    val tasks: ArrayBuffer[Future[_]] = new ArrayBuffer()
    var threshold = Long.MaxValue
    val timeout = Long.MaxValue
    var iteration = 0
    val dropPercentage = state.get[Double]("dropPercentage").get
    val warmupIterationNum = state.get[Int]("warmupIterationNum").get
    val computeThresholdbatchSize = state.get[Int]("computeThresholdBatchSize").get
    val maxDropPercentage = state.get[Double]("maxDropPercentage").get
    val driverSubModelNum = partitionNum * _subModelNumber
    var dropModelNumBatch = 0
    var lossArray = new Array[Double](_subModelNumber)

    var epochStart = System.nanoTime()
    var dataRDD = dataset.data(train = true)

    while (!endWhen(driverState)) {
      val lossSum = sc.doubleAccumulator("loss sum")
      val recordsNum = sc.doubleAccumulator("record number")
      val start = System.nanoTime()

      val numFinishedModelUpdates: Int = dataRDD.zipPartitions(models, preservesPartitioning = true) {
        (data, modelIter) => {
          val cached: CacheV1[T] = modelIter.next()
          val weightResults = parameters.getWeights(
            cached.modelWeights.head.narrow(0, parameters.paramOffset, parameters.size))

          val miniBatchBuffer = new Array[MiniBatch[T]](_subModelNumber)
          val batch: MiniBatch[T] = data.next()
          val stackSize = batch.size() / _subModelNumber
          tasks += Engine.default.invoke( () => {
            require(batch.size() >= _subModelNumber && batch.size() % _subModelNumber == 0)
            var b = 0
            while (b < _subModelNumber) {
              miniBatchBuffer(b) = batch.slice(b * stackSize, stackSize)
              b += 1
            }
          })
          Engine.default.sync(tasks)
          weightResults.waitResult()
          tasks.clear()

          // ====================== Start training models =============================
          val trainingThreads = Engine.default.invokeAndWait2((0 until _subModelNumber).map(i =>
            () => {
              val localModel: Module[T] = cached.localModels(i)
              localModel.training()
              val localCriterion = cached.localCriterions(i)
              val input = miniBatchBuffer(i).getInput
              val target = miniBatchBuffer(i).getTarget

              val output = localModel.forward(input).asInstanceOf[Tensor[T]]
              lossArray(i) = ev.toType[Double](localCriterion.forward(output, target))
              val errors = localCriterion.backward(output, target)
              localModel.backward(input, errors)
              i
            }
          ), timeout)

          val finishedThreads = trainingThreads.filter(!_.isCancelled).map(_.get())
          recordsNum.add(finishedThreads.size * stackSize)
          var i = 0
          while (i < finishedThreads.size) {
            lossSum.add(lossArray(finishedThreads(i)))
            i += 1
          }

          if (finishedThreads.nonEmpty) {
            val finishedGradients = finishedThreads.map(cached.modelGradients(_))
            val pOffset = parameters.paramOffset
            val pLength = parameters.size
            val taskSize = pLength / _subModelNumber
            val extraTask = pLength % _subModelNumber

            val parallelNum = if (taskSize == 0) extraTask else _subModelNumber
            if (parallelNum != 1) {
              Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
                val offset = pOffset + tid * taskSize + math.min(tid, extraTask)
                val length = taskSize + (if (tid < extraTask) 1 else 0)
                var i = 1
                while (i < finishedGradients.length) {
                  finishedGradients.head.narrow(0, offset, length)
                    .add(finishedGradients(i).narrow(0, offset, length))
                  i += 1
                }
              }))
            }
            // "finishedGradients" contains all gradients, which will be split to
            // different partitions in putGradients function
            parameters.putGradients(finishedGradients.head.narrow(0, pOffset, pLength))

          } else {
            cached.modelGradients(0).zero()
            parameters.putGradients(
              cached.modelGradients(0).narrow(0, parameters.paramOffset, parameters.size))
          }

          tasks ++= Engine.default.invoke {
            (0 until _subModelNumber).map { i => () => {
              cached.localModels(i).training()
              cached.localModels(i).zeroGradParameters()
            }}
          }
          Iterator.single(finishedThreads.size)
        }
      }.reduce(_ + _)

      dropModelNumBatch += (driverSubModelNum - numFinishedModelUpdates)
      if (dropPercentage == 0.0 ||
          numFinishedModelUpdates >= driverSubModelNum * (1.0 - maxDropPercentage)) {
        val value = lossSum.value / numFinishedModelUpdates
        driverState("numFinishedModel") = numFinishedModelUpdates
        driverState("isGradientUpdated") = false
        val isGradientUpdated = driverState[Boolean]("isGradientUpdated")
        val stateBroadcast = sc.broadcast(driverState)

        models.mapPartitions { modelIter =>
          val (paramLocalStart, paramLocalLen) = parameters.localPartitionRange
          val modelCache = modelIter.next()
          if (!isGradientUpdated) {
            parameters.aggregateGradientPartition(numFinishedModelUpdates)
          }
          // parameterProcessers.foreach(_.processParameters(parameters, modelCache, driverState))

          modelCache.optimMethods.foreach { case (name, optimMethod) =>
            optimMethod.state.update("epoch", driverState[Int]("epoch"))
            optimMethod.state.update("neval", driverState[Int]("neval"))
            optimMethod.state.update("Loss", driverState[Float]("Loss"))

            val p = parameterSplits(name)
            val startIdx = Math.max(paramLocalStart, p._1)
            val endIdx = Math.min(paramLocalStart + paramLocalLen, p._1 + p._2)
            if (endIdx > startIdx) {
              optimMethod.optimize(_ => (
                ev.fromType(value),
                parameters.gradientPartition.narrow(0, startIdx - paramLocalStart, endIdx - startIdx)
              ), parameters.weightPartition.narrow(0, startIdx - paramLocalStart, endIdx - startIdx))
            }
          }
          parameters.sendWeightPartition()
          Iterator.empty
        }.count()  // use count() to trigger mapPartitions transformation

        stateBroadcast.destroy()
        recordsProcessedThisEpoch += recordsNum.value.toInt
        val end = System.nanoTime()
        wallClockTime += end - start
        driverState("isGradientUpdated") = true
        driverState("Loss") = lossSum.value.toFloat / numFinishedModelUpdates
        optimMethods.foreach { v => v._2.updateHyperParameter()}

        driverState("LearningRate") = optimMethods.head._2.getLearningRate.toFloat
        driverState("Throughput") = recordsNum.value.toFloat / ((end - start) / 1e9f)
        lossArray = new Array[Double](_subModelNumber)

        iteration += 1
        driverState("neval") = driverState[Int]("neval") + 1
        if (recordsProcessedThisEpoch >= numSamples) {
          val epochEnd = System.nanoTime()
          wallClockTime = lastEpochTime + epochEnd - epochStart
          // logger.info(s" ... Epoch finished. Wall clock time is ${wallClockTime / 1e6} ms")
          lastEpochTime = wallClockTime
          epochStart = System.nanoTime()

          driverState("epoch") = driverState[Int]("epoch") + 1
          dataset.shuffle()
          dataRDD = dataset.data(train = true)
          recordsProcessedThisEpoch = 0
        }

        optimMethods.map { case (moduleName, optimMethod) =>
          optimMethod.state.update("recordsProcessedThisEpoch", recordsProcessedThisEpoch)
          optimMethod.state.update("epoch", driverState[Int]("epoch"))
          optimMethod.state.update("neval", driverState[Int]("neval"))
          optimMethod.state.update("Loss", driverState[Float]("Loss"))
        }

      } else {
        logger.info(s"Warning! Not enough training samples were successfully processed in this " +
          s"iteration due to some slow tasks. The gradients computed in this iteration will be " +
          s"discarded. Only $numFinishedModelUpdates/$driverSubModelNum threads successfully " +
          s"completed training.")
      }
    }
  }

  private def initThreadModels[T: ClassTag](
      model: Module[T],
      dataset: DistributedDataSet[MiniBatch[T]],
      criterion: Criterion[T],
      state: Table,
      nodeNumber: Int,
      coresPerNode: Int,
      // checkSingleton: Boolean,
      allReduceParameter: AllReduceParameter[T],
      parameterSplits: Map[String, (Int, Int)],
      // validationMethods: Option[Array[ValidationMethod[T]]],
      optimMethod: Map[String, OptimMethod[T]])(
      implicit ev: TensorNumeric[T]): (RDD[DistriOptimizer.CacheV1[T]], ModelBroadcast[T]) = {
    require(dataset.originRDD().partitions.length == nodeNumber,
      s"RDD partition number ${dataset.originRDD().partitions.length}" +
        s" is not equal to configured node number $nodeNumber")

    val sc = dataset.originRDD().sparkContext
    val broadcast = sc.broadcast((criterion, state, optimMethod))
    // val convertedModel = ConversionUtils.convert(model)
    val convertedModel = model
    convertedModel.adjustParameters()
    val modelBroadcast = ModelBroadcast[T]().broadcast(sc, convertedModel)
    val _subModelNumber = coresPerNode

    val computeThresholdbatchSize = state.get[Int]("computeThresholdBatchSize").get
    val nExecutor = Engine.nodeNumber()
    val executorCores = Engine.coreNumber()

    val models = dataset.originRDD().mapPartitions( _ => {
      val partitionId = TaskContext.getPartitionId
      val (broadcastCriterion, broadcastState, broadcastOptim) = broadcast.value
      Engine.setNodeAndCore(nExecutor, executorCores)
      // one model per core
      val cached = (0 until _subModelNumber).map { _ =>
        val localModel = modelBroadcast.value(initGradient = true, shareWeight = true)
        // setModelId(localModel, partitionId)
        localModel.setId(partitionId)
        val localCriterion = broadcastCriterion.cloneCriterion()
        val localState = broadcastState.clone()
        val (weights, grads) = localModel.adjustParameters()
        (localModel, weights, grads, localCriterion, localState)
      }.toArray

      logger.info("thread pool size is " + Engine.default.getPoolSize)
      // localModels are same, weights are shared
      val weights = cached.head._2
      allReduceParameter.init(weights.narrow(0, allReduceParameter.paramOffset,
        allReduceParameter.size))

      Iterator.single(CacheV1(
        cached.map(_._1), // models, one model per core
        cached.map(_._2), // weights
        cached.map(_._3), // gradients
        cached.map(_._4), // criterions
        cached.map(_._5), // states
        new Array[Long](_subModelNumber * computeThresholdbatchSize),
        //cached.map(_._6),
        broadcastOptim.map(v => (v._1, v._2.clone()))
      ))
    }).persist()

    models.setName("Thread Model RDD")
    logger.info("Cache thread models...")
    // to trigger persist
    models.count()
    logger.info("Cache thread models... done")
    (models, modelBroadcast)
  }

  override protected def getModel[T: ClassTag](
      models: RDD[Cache[T]],
      parameters: AllReduceParameter[T],
      trainingModel: Module[T])(implicit ev: TensorNumeric[T]): Module[T] = {
    val partitionNum = models.getNumPartitions
    Util.setExtraParametersFromModelRDD(models, trainingModel, maxSize = 5e8.toInt)

    val parameterArray = trainingModel.parameters()
    parameterArray._2.indices.foreach { i =>
      parameterArray._2(i).resizeAs(parameterArray._1(i))
    }
    val (parameter, gradientParameter) = trainingModel.adjustParameters()

    val (weights, gradients) = models.mapPartitions { _ =>
      val curPartitionId = TaskContext.getPartitionId()
      Iterator.single(
        (Map(curPartitionId -> parameters.weightPartition),
         Map(curPartitionId -> parameters.gradientPartition))
      )
    }.reduce((a, b) => (a._1 ++ b._1, a._2 ++ b._2))

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
}

class DistriOptimizer[T: ClassTag](
    _model: Module[T],
    _dataset: DistributedDataSet[MiniBatch[T]],
    _criterion: Criterion[T])(implicit ev: TensorNumeric[T])
    extends Optimizer[T, MiniBatch[T]](_model, _dataset, _criterion) {
  var compress: String = "fp16"
  private var models: RDD[DistriOptimizer.CacheV1[T]] = _
  private var modelBroadcast: ModelBroadcast[T] = _
  private var reserveOptimMethod = false
  private[sparkdl] var previousOptim: RDD[Map[String, OptimMethod[T]]] = _

  def setCompress(compressType: String): this.type = {
    compress = compressType
    this
  }

  def clearState(): Unit = {
    DistriOptimizer.clearState(models.asInstanceOf[RDD[DistriOptimizer.Cache[T]]])
  }

  override def reserveOptim(reserve: Boolean): DistriOptimizer.this.type = {
    reserveOptimMethod = reserve
    this
  }

  private def resetOptimMethods(
      model: RDD[DistriOptimizer.CacheV1[T]],
      previousOptimMethods: RDD[Map[String, OptimMethod[T]]])
      : RDD[DistriOptimizer.CacheV1[T]] = {
    models.zipPartitions(previousOptimMethods) { (m1, m2) =>
      val cache = m1.next()
      cache.optimMethods = m2.next()
      Iterator(cache)
    }
  }

  override def optimize(): Module[T] = {
    val distDataset = dataset.toDistributed
    val trainingModel = model
    optimMethods.values.foreach(_.clearHistory())
    if (optimMethods.size == 1) {
      optimMethods.head._2.loadFromTable(state)
    }

    state("dropPercentage") = dropPercentage
    state("warmupIterationNum") = warmupIterationNum
    state("computeThresholdBatchSize") = computeThresholdBatchSize
    state("maxDropPercentage") = maxDropPercentage

    val nodeNumber = Engine.nodeNumber()
    val coresPerNode = Engine.coreNumber()

    val partitionNum = distDataset.originRDD().partitions.length
    val modelParameters: (Tensor[T], Tensor[T]) = trainingModel.adjustParameters()
    val allReduceParameter = AllReduceParameter.newParameter[T](
      partitionNum, modelParameters._1.nElement(), compress = this.compress)
    val parameterSplits: Map[String, (Int, Int)] = {
      if (optimMethods.size != 1) {
        val p = optimMethods.map { case (subModuleName, optimMethod) =>
          val subModule = trainingModel(subModuleName)
          require(subModule.isDefined)
          val subModuleWeights = subModule.get.adjustParameters()._1
          (subModuleName, subModuleWeights)
        }
        val sortedWeights = p.values.toArray.sortWith((a, b) => a.storageOffset() < b.storageOffset())
        val compactWeights = Module.isCompact(sortedWeights)
        require(modelParameters._1 == compactWeights)
        p.map { case (subModuleName, weights) =>
          (subModuleName, (weights.storageOffset(), weights.nElement()))
        }
      } else if (optimMethods.contains(trainingModel.getName)) {
        Map(trainingModel.getName -> (0, modelParameters._1.nElement()))
      } else {
        throw new IllegalArgumentException(s"${trainingModel.getName} doesn't " +
          s"have corresponding OptimMethod")
      }
    }

    // prepareInput() todo

    val modelsAndBroadcast = DistriOptimizer.initThreadModels(
        trainingModel,
        distDataset,
        criterion,
        state,
        nodeNumber,
        coresPerNode,
        allReduceParameter,
        parameterSplits,
        optimMethods)

    models = {
      if (reserveOptimMethod && previousOptim != null) {
        resetOptimMethods(modelsAndBroadcast._1, previousOptim)
      } else {
        modelsAndBroadcast._1
      }
    }
    modelBroadcast = modelsAndBroadcast._2

    DistriOptimizer.optimize(
      trainingModel,
      distDataset,
      coresPerNode,
      state,
      endWhen,
      models,
      optimMethods,
      allReduceParameter,
      parameterSplits
    )

    DistriOptimizer.getModel(models.asInstanceOf[RDD[DistriOptimizer.Cache[T]]],
      allReduceParameter, trainingModel)

    clearState()
    shutdown()

    if (reserveOptimMethod) {
      previousOptim = models.map(m => m.optimMethods).cache()
      previousOptim.count()
    } else {
      if (previousOptim != null) {
        previousOptim.unpersist()
      }
    }
    models.unpersist()
    trainingModel
  }

  private def getLastestFile(path: String, fileName: String): String = {
    val fl = new java.io.File(path)
    val files = fl.listFiles(new FilenameFilter {
      override def accept(dir: File, name: String): Boolean = {
        name.startsWith(fileName)
      }
    })

    val last = files.reduce((a, b) => if (a.lastModified() > b.lastModified()) a else b)
    last.getPath
  }

  private[optim] override def shutdown(): Unit = {
    models.foreachPartition { iter =>
      iter.foreach(_.localModels.foreach(_.release()))
    }
    CachedModels.deleteKey(modelBroadcast.uuid)
  }
}
