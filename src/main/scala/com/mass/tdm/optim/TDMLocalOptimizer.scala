package com.mass.tdm.optim

import com.mass.sparkdl.{Criterion, Module}
import com.mass.sparkdl.optim.{OptimMethod, Trigger}
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.utils.{Engine, T, Table, Util}
import com.mass.tdm.dataset.{SampledMiniBatch, TDMDataSet, TDMLocalDataSet}
import org.apache.log4j.{Level, Logger}

class TDMLocalOptimizer(
      model: Module[Float],
      dataset: TDMLocalDataSet,
      criterion: Criterion[Float],
      optimizer: OptimMethod[Float],
      numIteration: Int,
      progressInterval: Int) {

  val logger: Logger = Logger.getLogger(getClass)
  // logger.setLevel(Level.INFO)

  private val workingModels: Array[Module[Float]] = {
    // to make the parameters compact
    model.adjustParameters()
    val wb: Array[Tensor[Float]] = Util.getAndClearWeightBias(model.parameters())

    // all models share same weight
    val subModelNum = Engine.coreNumber()
    val models: Array[Module[Float]] = (1 to subModelNum).toArray.map { i =>
      val m = model.cloneModule()
      Util.putWeightBias(wb, m)
      Util.initGradWeightBias(wb, m)
      m
    }
    Util.putWeightBias(wb, model)
    Util.initGradWeightBias(wb, model)
    models
  }

  private val state: Table = T()
  private val endWhen: Trigger = Trigger.maxIteration(numIteration, "trainIter")
  private val subModelNum = Engine.default.getPoolSize
  private var realParallelism: Int = -1
  private val (totalWeights, totalGradients) = model.adjustParameters()
  private val gradLength = totalGradients.nElement()
  private val syncGradTaskSize = gradLength / subModelNum
  private val syncGradExtraTask = gradLength % subModelNum
  private val syncGradParallelNum = if (syncGradTaskSize == 0) syncGradExtraTask else subModelNum
  private val workingGradients = workingModels.map(_.adjustParameters()._2)
  private val workingCriterion = (1 to subModelNum).map(_ => criterion.cloneCriterion()).toArray

  def optimize(): Module[Float] = {
    var wallClockTime = 0L
    var dataCount = 0
    optimizer.clearHistory()
    state("epoch") = state.get[Int]("epoch").getOrElse(1)
    state("trainIter") = state.get[Int]("trainIter").getOrElse(1)

    dataset.shuffle()
    val parallelConvert = dataset.parallelSampling
    var miniBatchIter = dataset.iteratorMiniBatch(train = true)
    while (!endWhen(state)) {
      val start = System.nanoTime()
      val batch: SampledMiniBatch = miniBatchIter.next()
      val miniBatchBuffer = convertBatch(batch, parallelConvert)
      val dataEndTime = System.nanoTime()

      val totalLoss = train(miniBatchBuffer)

      syncGradient()

      optimizer.updateState("epoch", state.get("epoch"))
      optimizer.updateState("trainIter", state.get("trainIter"))
      optimizer.optimize(_ => (totalLoss.toFloat, totalGradients), totalWeights)

      val end = System.nanoTime()
      wallClockTime += (end - start)
      dataCount += batch.getLength

      if (state[Int]("trainIter") % progressInterval == 0) {
        val header = f"[Epoch ${state[Int]("epoch")}, $dataCount/${dataset.size()}]" +
          f"[Iteration ${state[Int]("trainIter")}][Wall Clock Time: ${wallClockTime / 1e9d}%.4fs]"
        logger.info(f"$header \n" +
          f"loss: $totalLoss%.4f, iteration time: ${(end - start) / 1e9d}%.4fs \n" +
          f"data fetch time: ${(dataEndTime - start) / 1e9d}%.4fs, " +
          f"train time: ${(end - dataEndTime) / 1e9d}%.4fs. \n")
      }

      state("trainIter") = state[Int]("trainIter") + 1
      if (dataCount >= dataset.size()) {
        state("epoch") = state[Int]("epoch") + 1
        dataset.shuffle()
        miniBatchIter = dataset.iteratorMiniBatch(train = true)
        dataCount = 0
      }
    }
    shutdown()

    model
  }

  private def convertBatch(batch: SampledMiniBatch, parallel: Boolean)
      : Array[(Array[Tensor[Int]], Tensor[Float])] = {

    val allData = dataset.getData
    val miniBatchSize = if (parallel) batch.getLength else batch.expandedSize()
    val taskSize = miniBatchSize / subModelNum
    val extraSize = miniBatchSize % subModelNum
    realParallelism = if (taskSize == 0) extraSize else subModelNum
    val miniBatchBuffer = new Array[(Array[Tensor[Int]], Tensor[Float])](realParallelism)

    if (parallel) {
      Engine.default.invokeAndWait(
        (0 until realParallelism).map(i => () => {
          val offset = batch.getOffset + i * taskSize + math.min(i, extraSize)
          val length = taskSize + (if (i < extraSize) 1 else 0)
          miniBatchBuffer(i) = batch.convert(allData, offset, length, i)
        })
      )
    } else {
      batch.convertAll(allData)
      var i = 0
      while (i < realParallelism) {
        val offset = i * taskSize + math.min(i, extraSize)
        val length = taskSize + (if (i < extraSize) 1 else 0)
        miniBatchBuffer(i) = batch.slice(offset, length)
        i += 1
      }
    }

    miniBatchBuffer
  }

  def train(miniBatch: Array[(Array[Tensor[Int]], Tensor[Float])]): Double = {
    val lossSum = Engine.default.invokeAndWait(
      (0 until realParallelism).map(i => () => {
        val localModel = workingModels(i)
        localModel.zeroGradParameters()
        localModel.training()
        val localCriterion = workingCriterion(i)
        val (inputs, labels) = miniBatch(i)
        val outputs = localModel.forward(inputs.head).asInstanceOf[Tensor[Float]]
        val localLoss = localCriterion.forward(outputs, labels).toDouble
        val gradients = localCriterion.backward(outputs, labels)
        localModel.backward(inputs.head, gradients)
        localLoss
      })
    ).sum
    lossSum / realParallelism
  }

  private def syncGradient(): Unit = {
    Engine.default.invokeAndWait(
      (0 until syncGradParallelNum).map(tid => () => {
        val offset = tid * syncGradTaskSize + math.min(tid, syncGradExtraTask)
        val length = syncGradTaskSize + (if (tid < syncGradExtraTask) 1 else 0)
        var i = 0
        while (i < realParallelism) {
          if (i == 0) {
            totalGradients.narrow(0, offset, length)
              .copy(workingGradients(i).narrow(0, offset, length))
          } else {
            totalGradients.narrow(0, offset, length)
              .add(workingGradients(i).narrow(0, offset, length))
          }
          i += 1
        }
      })
    )
    totalGradients.div(realParallelism)
  }

  def shutdown(): Unit = {
    workingModels.foreach(_.release())
  }
}
