package com.mass.dr.optim

import java.text.DecimalFormat

import scala.collection.mutable.ListBuffer

import com.mass.dr.dataset.{LocalDataSet, MiniBatch}
import com.mass.dr.dataset.MiniBatch.{LayerTransformedBatch, RerankTransformedBatch}
import com.mass.dr.model.DeepRetrieval
import com.mass.dr.LayerModule
import com.mass.dr.evaluation.{EvalResult, Evaluator}
import com.mass.dr.loss.CrossEntropyLayer
import com.mass.sparkdl.nn.SampledSoftmaxLoss
import com.mass.sparkdl.optim.{Adam, OptimMethod}
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble
import com.mass.sparkdl.utils.{Engine, Table, Util}
import org.apache.log4j.Logger

class LocalOptimizer(
    dataset: LocalDataSet,
    drModel: DeepRetrieval,
    numIteration: Int,
    numLayer: Int,
    learningRate: Double,
    numSampled: Int,
    embedSize: Int,
    topk: Int,
    beamSize: Int,
    progressInterval: Int) {
  val logger: Logger = Logger.getLogger(getClass)

  val (layerModel, reRankModel) = (drModel.layerModel.model, drModel.reRankModel.model)
  val layerCriterion: CrossEntropyLayer = CrossEntropyLayer(numLayer)
  val layerOptimizer: OptimMethod[Double] = Adam(learningRate)
  val reRankCriterion: SampledSoftmaxLoss[Double] = SampledSoftmaxLoss(
    numSampled,
    dataset.numItem,
    embedSize,
    learningRate,
    drModel.reRankWeights,
    drModel.reRankBias,
    batchMode = false
  )
  val reRankOptimizer: OptimMethod[Double] = Adam(learningRate)

  val optimEvalResults: ListBuffer[EvalResult] = new ListBuffer[EvalResult]()
  private val state: Table = Table()
  private val subModelNum = Engine.coreNumber()
  private var realParallelism: Int = _

  lazy val layerCopiedModels: IndexedSeq[LayerModule[Double]] = {
    // make the parameters compact
    layerModel.adjustParameters()
    val weights: Array[Tensor[Double]] = Util.getAndClearWeightBias(layerModel.parameters())

    // all models share same weight
    val models = (1 to subModelNum).map { _ =>
      val m = layerModel.cloneModule()
      Util.putWeightBias(weights, m)
      Util.initGradWeightBias(weights, m)
      m
    }
    Util.putWeightBias(weights, layerModel)
    Util.initGradWeightBias(weights, layerModel)
    models
  }

  lazy val (totalLayerWeights, totalLayerGradients) = layerModel.adjustParameters()
  lazy val layerCopiedGradients = layerCopiedModels.map(_.adjustParameters()._2)
  lazy val layerCopiedCriterions = (1 to subModelNum).map(_ => layerCriterion.clone())

  def optimize(): Unit = {
    var epochTime = 0L
    var dataCount = 0
    layerOptimizer.clearHistory()
    reRankOptimizer.clearHistory()
    state("epoch") = state.get[Int]("epoch").getOrElse(0)
    state("trainIter") = state.get[Int]("trainIter").getOrElse(0)

    dataset.shuffle()
    var miniBatchIter = dataset.iteratorMiniBatch(train = true)
    0 until numIteration foreach { _ =>
      var start = System.nanoTime()
      val batch: MiniBatch = miniBatchIter.next()
      // train layer model
      val layerBatch = transformLayerBatch(batch)
      val layerLoss = trainLayerBatch(layerBatch)
      syncGradients()
      // use pseudo loss 0.0
      layerOptimizer.optimize(_ => (0.0, totalLayerGradients), totalLayerWeights)
      val layerTrainTime = System.nanoTime() - start

      // train rerank model
      start = System.nanoTime()
      val reRankLoss = trainRerank(batch)
      val (reRankWeights, reRankGradients) = reRankModel.adjustParameters()
      reRankOptimizer.optimize(_ => (reRankLoss, reRankGradients), reRankWeights)
      val reRankTrainTime = System.nanoTime() - start

      epochTime += (layerTrainTime + reRankTrainTime)
      dataCount += batch.getLength
      state("trainIter") = state[Int]("trainIter") + 1
      if (progressInterval > 0 && state[Int]("trainIter") % progressInterval == 0) {
        reportProgress(dataCount, layerTrainTime, reRankTrainTime, epochTime, layerLoss, reRankLoss)
      }
      if (dataCount >= dataset.trainSize) {
        reportProgress(dataCount, layerTrainTime, reRankTrainTime, epochTime, layerLoss, reRankLoss)
        state("epoch") = state[Int]("epoch") + 1
        dataset.shuffle()
        miniBatchIter = dataset.iteratorMiniBatch(train = true)
        epochTime = 0L
        dataCount = 0
      }
    }
    shutdown()
  }

  private def trainRerank(batch: MiniBatch): Double = {
    val reRankBatch: RerankTransformedBatch = batch.transformRerankData(
      dataset.getTrainData, batch.getOffset, batch.getLength
    )
    val inputs = reRankBatch.itemSeqs
    val labels = reRankBatch.target.asInstanceOf[Tensor[Double]]
    reRankModel.zeroGradParameters()
    reRankModel.training()
    val outputs = reRankModel.forward(inputs).toTensor[Double]
    val reRankLoss = reRankCriterion.forward(outputs, labels)
    val lossGrad = reRankCriterion.backward(outputs, labels)
    reRankModel.backward(inputs, lossGrad)
    reRankLoss
  }

  private def transformLayerBatch(batch: MiniBatch): Seq[LayerTransformedBatch] = {
    val allData = dataset.getTrainData
    val miniBatchOffset = batch.getOffset
    val miniBatchSize = batch.getLength
    val taskSize = miniBatchSize / subModelNum
    val extraSize = miniBatchSize % subModelNum
    realParallelism = if (taskSize == 0) extraSize else subModelNum
    Engine.default.invokeAndWait(
      (0 until realParallelism).map(i => () => {
        val offset = miniBatchOffset + i * taskSize + math.min(i, extraSize)
        val length = taskSize + (if (i < extraSize) 1 else 0)
        batch.transformLayerData(allData, offset, length)
      })
    )
  }

  private def trainLayerBatch(miniBatch: Seq[LayerTransformedBatch]): Seq[Double] = {
    val lossSum = Engine.default.invokeAndWait(
      (0 until realParallelism).map(i => () => {
        val localModel = layerCopiedModels(i)
        localModel.zeroGradParameters()
        localModel.training()
        val localCriterion = layerCopiedCriterions(i)
        val inputs = Table.seq(miniBatch(i).concatInputs)
        val labels = miniBatch(i).targets
        val outputs = localModel.forward(inputs).toTable
        val localLoss = localCriterion.forward(outputs, labels)
        val gradients = localCriterion.backward(outputs, labels)
        localModel.backward(inputs, gradients)
        localLoss
      })
    ).reduce((a, b) => a.lazyZip(b).map(_ + _))
    lossSum.map(_ / realParallelism)
  }

  private def syncGradients(): Unit = {
    val gradLength = totalLayerGradients.nElement()
    val syncGradTaskSize = gradLength / subModelNum
    val syncGradExtraSize = gradLength % subModelNum
    val syncGradParallelNum = if (syncGradTaskSize == 0) syncGradExtraSize else subModelNum
    Engine.default.invokeAndWait(
      (0 until syncGradParallelNum).map(tid => () => {
        val offset = tid * syncGradTaskSize + math.min(tid, syncGradExtraSize)
        val length = syncGradTaskSize + (if (tid < syncGradExtraSize) 1 else 0)
        List.range(0, realParallelism).foreach {
          case i@0 =>
            totalLayerGradients.narrow(0, offset, length)
              .copy(layerCopiedGradients(i).narrow(0, offset, length))
          case i@_ =>
            totalLayerGradients.narrow(0, offset, length)
              .add(layerCopiedGradients(i).narrow(0, offset, length))
        }
      })
    )
    totalLayerGradients.div(realParallelism)
  }

  private def reportProgress(
      dataCount: Int,
      layerTrainTime: Long,
      reRankTrainTime: Long,
      epochTime: Long,
      layerLoss: Seq[Double],
      reRankLoss: Double): Unit = {
    val formatter = new DecimalFormat("##.####")
    val progressInfo = new StringBuilder
    val iterationTime = layerTrainTime + reRankTrainTime
    progressInfo ++= f"Epoch ${state[Int]("epoch") + 1} train time: ${epochTime / 1e9d}%.4fs, "
    progressInfo ++= f"count/total: $dataCount/${dataset.trainSize}, "
    progressInfo ++= f"Iteration ${state[Int]("trainIter")} time: ${iterationTime / 1e9d}%.4fs\n"
    progressInfo ++= f"\t\tlayer iteration train time: ${layerTrainTime / 1e9d}%.4f, "
    progressInfo ++= f"rerank iteration train time: ${reRankTrainTime / 1e9d}%.4f\n"
    progressInfo ++= f"\t\ttrain layer loss: ${layerLoss.map(formatter.format).mkString("[",", ","]")}, "
    progressInfo ++= f"train rerank loss: $reRankLoss%.4f\n"

    val evalStart = System.nanoTime()
    val evalResult = Evaluator.evaluate(
      dataset,
      drModel,
      layerCriterion,
      reRankCriterion,
      embedSize,
      topk,
      beamSize
    )
    val evalEnd = System.nanoTime()
    optimEvalResults += evalResult
    progressInfo ++= f"\teval time: ${(evalEnd - evalStart) / 1e9d}%.4fs, Metrics: $evalResult\n"
    logger.info(progressInfo.toString)
  }

  def shutdown(): Unit = {
    layerCopiedModels.foreach(_.release())
  }
}
