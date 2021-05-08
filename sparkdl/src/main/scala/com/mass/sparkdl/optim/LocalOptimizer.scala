package com.mass.sparkdl.optim

import scala.reflect.ClassTag

import com.mass.sparkdl.{Criterion, Module}
import com.mass.sparkdl.dataset.{LocalDataSet, MiniBatch}
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}
import com.mass.sparkdl.utils.{Engine, Util}
import org.apache.log4j.Logger

class LocalOptimizer[T: ClassTag](
    model: Module[T],
    dataset: LocalDataSet[MiniBatch[T]],
    criterion: Criterion[T],
    optimizer: OptimMethod[T])(implicit ev: TensorNumeric[T])
    extends Optimizer[T, MiniBatch[T]](model, dataset, criterion) {

  import LocalOptimizer.logger
  import Optimizer.header

  private val coreNumber = Engine.coreNumber()
  private val subModelNumber = coreNumber

  private val workingModels: Array[Module[T]] = {
    // to make the parameters compact
    model.adjustParameters()
    val wb: Array[Tensor[T]] = Util.getAndClearWeightBias(model.parameters())

    val models: Array[Module[T]] = (1 to subModelNumber).map { i =>
      val m = model.cloneModule()
      Util.putWeightBias(wb, m)
      Util.initGradWeightBias(wb, m)
      m
    }.toArray
    Util.putWeightBias(wb, model)
    Util.initGradWeightBias(wb, model)
    models
  }

  private val (weight, grad) = model.adjustParameters()
  private val gradLength = grad.nElement()
  private val syncGradTaskSize = gradLength / subModelNumber
  private val syncGradExtraTask = gradLength % subModelNumber
  private val syncGradParallelNum = if (syncGradTaskSize == 0) syncGradExtraTask else subModelNumber
  private val workingModelWAndG = workingModels.map(_.adjustParameters())
  private val workingCriterion = (1 to subModelNumber).map(_ => criterion.cloneCriterion()).toArray

  override def optimize(): Module[T] = {
    var wallClockTime = 0L
    var count = 0
    optimizer.clearHistory()
    optimizer.loadFromTable(state)
    state("epoch") = state.get[Int]("epoch").getOrElse(1)
    state("neval") = state.get[Int]("neval").getOrElse(1)

    dataset.shuffle()
    val numSamples = dataset.data(train = false).map(_.size()).sum
    var iter = dataset.data(train = true)
    logger.info("Thread pool size is " + Engine.default.getPoolSize)
    while (!endWhen(state)) {
      val start = System.nanoTime()
      val batch: MiniBatch[T] = iter.next()
      val stackSize = batch.size() / subModelNumber
      val extraSize = batch.size() % subModelNumber
      val parallelism = if (stackSize == 0) extraSize else subModelNumber
      state("parallelism") = parallelism
      val miniBatchBuffer = new Array[MiniBatch[T]](parallelism)
      var b = 0
      while (b < parallelism) {
        val offset = b * stackSize + math.min(b, extraSize)
        val length = stackSize + (if (b < extraSize) 1 else 0)
        miniBatchBuffer(b) = batch.slice(offset, length)
        b += 1
      }
      val dataFetchTime = System.nanoTime()

      val lossSum = Engine.default.invokeAndWait(
        (0 until parallelism).map(i => () => {
          val localModel = workingModels(i)
          localModel.zeroGradParameters()
          localModel.training()
          val localCriterion = workingCriterion(i)
          val input = miniBatchBuffer(i).getInput
          val target = miniBatchBuffer(i).getTarget
          val output = localModel.forward(input).asInstanceOf[Tensor[T]]
          val localLoss = ev.toType[Double](localCriterion.forward(output, target))
          val errors = localCriterion.backward(output, target)
          localModel.backward(input, errors)
          localLoss
        })
      ).sum
      val loss = lossSum / parallelism

      Engine.default.invokeAndWait(
        (0 until syncGradParallelNum).map(tid => () => {
          val offset = tid * syncGradTaskSize + math.min(tid, syncGradExtraTask)
          val length = syncGradTaskSize + (if (tid < syncGradExtraTask) 1 else 0)
          var i = 0
          while (i < syncGradParallelNum) {
            if (i == 0) {
              grad.narrow(0, offset, length).copy(workingModelWAndG(i)._2.narrow(0, offset, length))
            } else {
              grad.narrow(0, offset, length).add(workingModelWAndG(i)._2.narrow(0, offset, length))
            }
            i += 1
          }
        })
      )
      grad.div(ev.fromType(syncGradParallelNum))

      optimizer.state.update("epoch", state.get("epoch"))
      optimizer.state.update("neval", state.get("neval"))
      optimizer.optimize(_ => (ev.fromType(loss), grad), weight)

      val end = System.nanoTime()
      wallClockTime += (end - start)
      count += batch.size()
      val head = header(state[Int]("epoch"), count, numSamples, state[Int]("neval"), wallClockTime)
      state("neval") = state[Int]("neval") + 1
      logger.info(s"$head \n" +
        s"loss: $loss, iteration time: ${(end - start) / 1e9}s \n" +
        s"data fetch time: ${(dataFetchTime - start) / 1e9}s, " +
        s"train time: ${(end - dataFetchTime) / 1e9}s. \n" +
        s"Throughput is ${batch.size().toDouble / (end - start) * 1e9} record / second.\n")

      if (count >= numSamples) {
        state("epoch") = state[Int]("epoch") + 1
        dataset.shuffle()
        iter = dataset.toLocal.data(train = true)
        count = 0
      }
    }

    model.setExtraParameter(workingModels.head.getExtraParameter)
    shutdown()

    model
  }

  private[optim] override def shutdown(): Unit = {
    workingModels.foreach(_.release())
  }
}

object LocalOptimizer {
  val logger: Logger = Logger.getLogger(getClass)
}
