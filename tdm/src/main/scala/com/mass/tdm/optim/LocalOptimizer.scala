package com.mass.tdm.optim

import scala.collection.mutable

import com.mass.scalann.{Criterion, Module}
import com.mass.scalann.optim.{OptimMethod, Trigger}
import com.mass.scalann.tensor.Tensor
import com.mass.scalann.tensor.TensorNumeric.NumericFloat
import com.mass.scalann.utils.{Engine, Table, Util}
import com.mass.tdm.dataset.{LocalDataSet, MiniBatch}
import com.mass.tdm.dataset.MiniBatch.{MaskTransformedBatch, SeqTransformedBatch, TransformedBatch}
import com.mass.tdm.evaluation.Evaluator
import org.apache.log4j.Logger

class LocalOptimizer(
    model: Module[Float],
    dataset: LocalDataSet,
    criterion: Criterion[Float],
    optimMethod: OptimMethod[Float],
    numIteration: Int,
    progressInterval: Int,
    topk: Int,
    candidateNum: Int,
    useMask: Boolean
) {
  val logger: Logger = Logger.getLogger(getClass)

  private val workingModels: Array[Module[Float]] = {
    // to make the parameters compact
    model.adjustParameters()
    val wb: Array[Tensor[Float]] = Util.getAndClearWeightBias(model.parameters())

    // all models share same weight
    val subModelNum = Engine.coreNumber()
    val models: Array[Module[Float]] = (1 to subModelNum).toArray.map { _ =>
      val m = model.cloneModule()
      Util.putWeightBias(wb, m)
      Util.initGradWeightBias(wb, m)
      m
    }
    Util.putWeightBias(wb, model)
    Util.initGradWeightBias(wb, model)
    models
  }

  private val state: Table = Table()
  private val endWhen: Trigger = Trigger.maxIteration(numIteration, "trainIter")
  private val subModelNum = Engine.coreNumber()
  private var realParallelism: Int = -1
  private val (totalWeights, totalGradients) = model.adjustParameters()
  private val gradLength = totalGradients.nElement()
  private val syncGradTaskSize = gradLength / subModelNum
  private val syncGradExtraTask = gradLength % subModelNum
  private val syncGradParallelNum = if (syncGradTaskSize == 0) syncGradExtraTask else subModelNum
  private val workingGradients = workingModels.map(_.adjustParameters()._2)
  private val workingCriterions = (1 to subModelNum).map(_ => criterion.cloneCriterion()).toArray

  def optimize(): Module[Float] = {
    var epochTime = 0L
    var dataCount = 0
    optimMethod.clearHistory()
    state("epoch") = state.get[Int]("epoch").getOrElse(0)
    state("trainIter") = state.get[Int]("trainIter").getOrElse(0)

    dataset.shuffle()
    var miniBatchIter = dataset.iteratorMiniBatch(train = true, expandBatch = true)
    while (!endWhen(state)) {
      val start = System.nanoTime()
      val batch: MiniBatch = miniBatchIter.next()
      val miniBatchBuffer = convertBatch(batch)
      val loss = trainBatch(miniBatchBuffer)

      syncGradients()
      optimMethod.optimize(_ => (loss.toFloat, totalGradients), totalWeights)
      val end = System.nanoTime()
      val iterationTime = end - start
      epochTime += iterationTime
      dataCount += batch.getLength
      state("trainIter") = state[Int]("trainIter") + 1

      if (progressInterval > 0 && state[Int]("trainIter") % progressInterval == 0) {
        reportProgress(
          dataset,
          workingModels,
          workingCriterions,
          topk,
          candidateNum,
          state,
          dataCount,
          iterationTime,
          epochTime,
          loss
        )
      }

      if (dataCount >= dataset.trainSize) {
        reportProgress(
          dataset,
          workingModels,
          workingCriterions,
          topk,
          candidateNum,
          state,
          dataCount,
          iterationTime,
          epochTime,
          loss
        )
        state("epoch") = state[Int]("epoch") + 1
        dataset.shuffle()
        miniBatchIter = dataset.iteratorMiniBatch(train = true, expandBatch = true)
        epochTime = 0L
        dataCount = 0
      }
    }

    updateOptimState(state, optimMethod)
    shutdown()
    model
  }

  private def convertBatch(batch: MiniBatch): Array[TransformedBatch] = {
    val allData = dataset.getData
    val miniBatchSize = batch.getLength
    val taskSize = miniBatchSize / subModelNum
    val extraSize = miniBatchSize % subModelNum
    realParallelism = if (taskSize == 0) extraSize else subModelNum
    val miniBatchBuffer = new Array[TransformedBatch](realParallelism)
    Engine.default.invokeAndWait(
      (0 until realParallelism).map { i => () =>
        val offset = batch.getOffset + i * taskSize + math.min(i, extraSize)
        val length = taskSize + (if (i < extraSize) 1 else 0)
        miniBatchBuffer(i) = batch.convert(allData, offset, length, i)
      }
    )
    miniBatchBuffer
  }

  def trainBatch(miniBatch: Array[TransformedBatch]): Double = {
    val lossSum = Engine.default
      .invokeAndWait(
        (0 until realParallelism).map { i => () =>
          val localModel = workingModels(i)
          localModel.zeroGradParameters()
          localModel.training()
          val localCriterion = workingCriterions(i)
          val (inputs, labels) = miniBatch(i) match {
            case m: SeqTransformedBatch =>
              (Table(m.items, m.sequence), m.labels)
            case m: MaskTransformedBatch =>
              (Table(m.items, m.sequence, m.masks), m.labels)
          }
          val outputs = localModel.forward(inputs).toTensor
          val localLoss = localCriterion.forward(outputs, labels).toDouble
          val gradients = localCriterion.backward(outputs, labels)
          localModel.backward(inputs, gradients)
          localLoss
        }
      )
      .sum
    lossSum / realParallelism
  }

  private def syncGradients(): Unit = {
    Engine.default.invokeAndWait(
      (0 until syncGradParallelNum).map(tid =>
        () => {
          val offset = tid * syncGradTaskSize + math.min(tid, syncGradExtraTask)
          val length = syncGradTaskSize + (if (tid < syncGradExtraTask) 1 else 0)
          var i = 0
          while (i < realParallelism) {
            if (i == 0) {
              totalGradients
                .narrow(0, offset, length)
                .copy(workingGradients(i).narrow(0, offset, length))
            } else {
              totalGradients
                .narrow(0, offset, length)
                .add(workingGradients(i).narrow(0, offset, length))
            }
            i += 1
          }
        }
      )
    )
    totalGradients.div(realParallelism.toFloat)
  }

  private def updateOptimState(state: Table, optimMethod: OptimMethod[Float]): Unit = {
    optimMethod.state.update("epoch", state[Int]("epoch"))
    optimMethod.state.update("trainIter", state[Int]("trainIter"))
  }

  def shutdown(): Unit = {
    workingModels.foreach(_.release())
  }

  private def reportProgress(
      dataset: LocalDataSet,
      models: Array[Module[Float]],
      criterions: Array[Criterion[Float]],
      topk: Int,
      candidateNum: Int,
      state: Table,
      dataCount: Int,
      iterationTime: Long,
      epochTime: Long,
      trainLoss: Double
  ): Unit = {
    val progressInfo = new mutable.StringBuilder
    progressInfo ++= f"Epoch ${state[Int]("epoch") + 1} train time: ${epochTime / 1e9d}%.4fs, "
    progressInfo ++= s"count/total: $dataCount/${dataset.trainSize}, "
    progressInfo ++= f"Iteration ${state[Int]("trainIter")} time: ${iterationTime / 1e9d}%.4fs, "
    progressInfo ++= f"Train loss: $trainLoss%.4f\n"

    val evalStart = System.nanoTime()
    val evalResult = Evaluator.evaluate(
      models,
      dataset,
      criterions,
      topk,
      candidateNum,
      useMask
    )
    val evalEnd = System.nanoTime()
    progressInfo ++= f"\teval time: ${(evalEnd - evalStart) / 1e9d}%.4fs, Metrics: $evalResult\n"
    logger.info(progressInfo.toString)
  }
}

object LocalOptimizer {

  def apply(
      model: Module[Float],
      dataset: LocalDataSet,
      criterion: Criterion[Float],
      optimMethod: OptimMethod[Float],
      numIteration: Int,
      progressInterval: Int,
      topk: Int,
      candidateNum: Int,
      useMask: Boolean
  ): LocalOptimizer = {
    new LocalOptimizer(
      model,
      dataset,
      criterion,
      optimMethod,
      numIteration,
      progressInterval,
      topk,
      candidateNum,
      useMask
    )
  }
}
