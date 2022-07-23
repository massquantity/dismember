package com.mass.dr.optim

import java.text.DecimalFormat

import com.mass.dr.dataset.{LocalDataSet, MiniBatch}
import com.mass.dr.model.{LayerModel, RerankModel}
import com.mass.dr.evaluation.{EvalResult, Evaluator}
import com.mass.dr.loss.CrossEntropyLayer
import com.mass.dr.Path
import com.mass.sparkdl.nn.SampledSoftmaxLoss
import com.mass.sparkdl.optim.{Adam, OptimMethod}
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble
import com.mass.sparkdl.utils.{Engine, Table}
import org.apache.log4j.Logger

class LocalOptimizer(
    dataset: LocalDataSet,
    layerModel: LayerModel,
    reRankModel: RerankModel,
    numEpoch: Int,
    numLayer: Int,
    learningRate: Double,
    numSampled: Int,
    embedSize: Int,
    topk: Int,
    beamSize: Int,
    progressInterval: Int,
    reRankEpoch: Option[Int]) {
  val logger: Logger = Logger.getLogger(getClass)

  val reRankStoppingEpoch = reRankEpoch match {
    case Some(i) => i
    case None => numEpoch
  }
  val layerCriterion: CrossEntropyLayer = CrossEntropyLayer(numLayer)
  val layerOptimizer: OptimMethod[Double] = Adam(learningRate)
  val reRankCriterion: SampledSoftmaxLoss[Double] = SampledSoftmaxLoss(
    numSampled,
    dataset.numItem,
    embedSize,
    learningRate,
    reRankModel.softmaxWeights,
    reRankModel.softmaxBiases,
    batchMode = false
  )
  val reRankOptimizer: OptimMethod[Double] = Adam(learningRate)

  private val numThread = Engine.coreNumber()
  private val itemPathMapping = dataset.itemPathMapping

  val (layerCopiedModels, layerCopiedGradients) = layerModel.duplicateModels(numThread)
  val layerCopiedCriterions = (1 to numThread).map(_ => layerCriterion.clone())

  def optimize(): Vector[EvalResult] = {
    (1 to numEpoch).foldLeft(Vector.empty[EvalResult]) { (epochResults, epoch) =>
      dataset.shuffle()
      var epochTime, dataCount, iter = 0L
      val miniBatchIter = dataset.iteratorMiniBatch(train = true)
      while (miniBatchIter.hasNext) {
        val layerStart = System.nanoTime()
        val batch = miniBatchIter.next()
        // train layer model
        val (layerLoss, parallelism) = trainLayerBatch(batch, itemPathMapping)
        val (totalLayerWeights, totalLayerGradients) = syncGradients(parallelism)
        layerOptimizer.optimize(_ => (0.0, totalLayerGradients), totalLayerWeights)
        val layerTrainTime = System.nanoTime() - layerStart

        // train rerank model
        val (reRankLoss, reRankTrainTime) =
          if (epoch <= reRankStoppingEpoch) {
            val reRankStart = System.nanoTime()
            val reRankLoss = trainRerank(batch)
            val (reRankWeights, reRankGradients) = reRankModel.parameters
            reRankOptimizer.optimize(_ => (reRankLoss, reRankGradients), reRankWeights)
            (reRankLoss, System.nanoTime() - reRankStart)
          } else {
            (Double.NaN, 0L)
          }

        epochTime += (layerTrainTime + reRankTrainTime)
        dataCount += batch.getLength
        iter += 1
        if (iter % progressInterval == 0 || !miniBatchIter.hasNext) {
          if (epoch > reRankStoppingEpoch) {
            logger.info(s"Rerank training stopped in epoch $epoch")
          }
          val iterMetrics = reportProgress(
            epoch,
            iter,
            dataCount,
            layerTrainTime,
            reRankTrainTime,
            epochTime,
            layerLoss,
            reRankLoss,
            itemPathMapping
          )
          logger.info(iterMetrics)
        }
      }
      epochResults :+ Evaluator.evaluate(
        dataset,
        layerModel,
        reRankModel,
        layerCriterion,
        reRankCriterion,
        topk,
        beamSize,
        itemPathMapping
      )
    }
  }

  private def trainRerank(batch: MiniBatch): Double = {
    val reRankBatch = batch.transformRerankData(
      dataset.getTrainData,
      batch.getOffset,
      batch.getLength
    )
    val inputs = reRankBatch.itemSeqs
    val labels = reRankBatch.target.asInstanceOf[Tensor[Double]]
    reRankModel.model.zeroGradParameters()
    reRankModel.model.training()
    val outputs = reRankModel.model.forward(inputs).toTensor[Double]
    val reRankLoss = reRankCriterion.forward(outputs, labels)
    val lossGrad = reRankCriterion.backward(outputs, labels)
    reRankModel.model.backward(inputs, lossGrad)
    reRankLoss
  }

  private def trainLayerBatch(
    batch: MiniBatch,
    itemPathMapping: Map[Int, Seq[Path]]
  ): (Seq[Double], Int) = {
    val allData = dataset.getTrainData
    val miniBatchOffset = batch.getOffset
    val miniBatchSize = batch.getLength
    val taskSize = miniBatchSize / numThread
    val extraSize = miniBatchSize % numThread
    val parallelism = if (taskSize == 0) extraSize else numThread
    val lossSum = Engine.default.invokeAndWait(
      (0 until parallelism).map { i => () =>
        val offset = miniBatchOffset + i * taskSize + math.min(i, extraSize)
        val length = taskSize + (if (i < extraSize) 1 else 0)
        val layerBatch = batch.transformLayerData(allData, offset, length, itemPathMapping)
        val localModel = layerCopiedModels(i)
        localModel.zeroGradParameters()
        localModel.training()
        val localCriterion = layerCopiedCriterions(i)
        val inputs = Table.seq(layerBatch.concatInputs)
        val labels = layerBatch.targets
        val outputs = localModel.forward(inputs).toTable
        val localLoss = localCriterion.forward(outputs, labels)
        val gradients = localCriterion.backward(outputs, labels)
        localModel.backward(inputs, gradients)
        localLoss
      }
    ).reduce((a, b) => a.lazyZip(b).map(_ + _))
    (lossSum.map(_ / parallelism), parallelism)
  }

  private def syncGradients(syncNum: Int): (Tensor[Double], Tensor[Double]) = {
    val (totalLayerWeights, totalLayerGradients) = layerModel.parameters
    val gradLength = totalLayerGradients.nElement()
    val syncGradTaskSize = gradLength / numThread
    val syncGradExtraSize = gradLength % numThread
    val syncGradParallelNum = if (syncGradTaskSize == 0) syncGradExtraSize else numThread
    Engine.default.invokeAndWait(
      (0 until syncGradParallelNum).map(tid => () => {
        val offset = tid * syncGradTaskSize + math.min(tid, syncGradExtraSize)
        val length = syncGradTaskSize + (if (tid < syncGradExtraSize) 1 else 0)
        List.range(0, syncNum).foreach {
          case i@0 =>
            totalLayerGradients.narrow(0, offset, length)
              .copy(layerCopiedGradients(i).narrow(0, offset, length))
          case i@_ =>
            totalLayerGradients.narrow(0, offset, length)
              .add(layerCopiedGradients(i).narrow(0, offset, length))
        }
      })
    )
    totalLayerGradients.div(syncNum)
    (totalLayerWeights, totalLayerGradients)
  }

  private def reportProgress(
    epoch: Int,
    iteration: Long,
    dataCount: Long,
    layerTrainTime: Long,
    reRankTrainTime: Long,
    epochTime: Long,
    layerLoss: Seq[Double],
    reRankLoss: Double,
    itemPathMapping: Map[Int, Seq[Path]]
  ): String = {
    val formatter = new DecimalFormat("##.####")
    val progressInfo = new StringBuilder
    val iterationTime = layerTrainTime + reRankTrainTime
    progressInfo ++= f"Epoch $epoch train time: ${epochTime / 1e9d}%.4fs, "
    progressInfo ++= f"count/total: $dataCount/${dataset.trainSize}, "
    progressInfo ++= f"Iteration $iteration time: ${iterationTime / 1e9d}%.4fs\n"
    progressInfo ++= f"\t\tlayer iteration train time: ${layerTrainTime / 1e9d}%.4f, "
    progressInfo ++= f"rerank iteration train time: ${reRankTrainTime / 1e9d}%.4f\n"
    progressInfo ++= f"\t\ttrain layer loss: ${layerLoss.map(formatter.format).mkString("[",", ","]")}, "
    progressInfo ++= f"train rerank loss: $reRankLoss%.4f\n"
    progressInfo ++= s"\t${evaluateStr(itemPathMapping)}"
    progressInfo.toString
  }

  def evaluateStr(itemPathMapping: Map[Int, Seq[Path]]): String = {
    val evalStart = System.nanoTime()
    val evalResult = Evaluator.evaluate(
      dataset,
      layerModel,
      reRankModel,
      layerCriterion,
      reRankCriterion,
      topk,
      beamSize,
      itemPathMapping
    )
    val evalEnd = System.nanoTime()
    f"eval time: ${(evalEnd - evalStart) / 1e9d}%.4fs, Metrics: $evalResult\n"
  }
}

object LocalOptimizer {

  def apply(
    dataset: LocalDataSet,
    layerModel: LayerModel,
    reRankModel: RerankModel,
    numEpoch: Int,
    numLayer: Int,
    learningRate: Double,
    numSampled: Int,
    embedSize: Int,
    topk: Int,
    beamSize: Int,
    progressInterval: Int,
    reRankEpoch: Option[Int] = None,
  ): LocalOptimizer = {
    new LocalOptimizer(
      dataset,
      layerModel,
      reRankModel,
      numEpoch,
      numLayer,
      learningRate,
      numSampled,
      embedSize,
      topk,
      beamSize,
      progressInterval,
      reRankEpoch
    )
  }
}
