package com.mass.dr.evaluation

import com.mass.dr.dataset.LocalDataSet
import com.mass.dr.evaluation.Metrics.computeMetrics
import com.mass.dr.model.{CandidateSearcher, DeepRetrieval, LayerModel, RerankModel}
import com.mass.dr.dataset.MiniBatch.{LayerTransformedBatch, RerankTransformedBatch}
import com.mass.dr.loss.CrossEntropyLayer
import com.mass.sparkdl.nn.SampledSoftmaxLoss
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.utils.Engine

object Evaluator extends Serializable with CandidateSearcher {

  def evaluate(
    dataset: LocalDataSet,
    drModel: DeepRetrieval,
    layerCriterion: CrossEntropyLayer,
    reRankCriterion: SampledSoftmaxLoss[Double],
    embedSize: Int,
    topk: Int,
    beamSize: Int
  ): EvalResult = {
    val subModelNum = Engine.coreNumber()
    val userConsumed = dataset.getUserConsumed
    val allData = dataset.getEvalData
    val (layerModel, reRankModel) = (drModel.layerModel, drModel.reRankModel)
    val miniBatchIter = dataset.iteratorMiniBatch(train = false)
    val evalResults = miniBatchIter.map { batch =>
      val offset = batch.getOffset
      val length = batch.getLength
      val layerMiniBatch = batch.transformLayerData(allData, offset, length)
      val layerLoss = evaluateLayerModel(layerMiniBatch, layerModel, layerCriterion)
      val reRankBatch = batch.transformRerankData(allData, offset, length)
      val reRankLoss = evaluateReRankModel(reRankBatch, reRankModel, reRankCriterion)

      val miniBatchSize = batch.getLength
      val taskSize = miniBatchSize / subModelNum
      val extraSize = miniBatchSize % subModelNum
      val realParallelism = if (taskSize == 0) extraSize else subModelNum
      val (precision, recall, ndcg) = Engine.default.invokeAndWait(
        0 until realParallelism map { i => () =>
          val offset = batch.getOffset + i * taskSize + math.min(i, extraSize)
          val length = taskSize + (if (i < extraSize) 1 else 0)
          List.range(offset, offset + length).foldLeft((0.0, 0.0, 0.0)) { case (m1, j) =>
            val data = allData(j)
            val consumedItems = userConsumed(data.user).toSet
            val labels = data.labels
            val recItems = recommendItems(drModel, data.sequence, embedSize, topk, beamSize, consumedItems)
            val m2 = computeMetrics(recItems, labels)
            (m1._1 + m2._1, m1._2 + m2._2, m1._3 + m2._3)
          }
        }
      ).reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
      EvalResult(layerLoss, reRankLoss, precision, recall, ndcg, length)
    }
    evalResults.reduce(_ + _)
  }

  private def evaluateLayerModel(
    batch: LayerTransformedBatch,
    layerModel: LayerModel,
    layerCriterion: CrossEntropyLayer
  ): Seq[Double] = {
    val inputs = batch.concatInputs.map(_.asInstanceOf[Tensor[Double]])
    val labels = batch.targets
    val outputs = layerModel.forward(inputs)
    val loss = layerCriterion.forward(outputs, labels)
    loss
  }

  private def evaluateReRankModel(
    batch: RerankTransformedBatch,
    reRankModel: RerankModel,
    reRankCriterion: SampledSoftmaxLoss[Double]
  ): Double = {
    val inputs = batch.itemSeqs.asInstanceOf[Tensor[Double]]
    val tmp = batch.target.storage().array().map(_.toDouble)
    val labels = Tensor(tmp, Array(tmp.length, 1))
    val outputs = reRankModel.forward(inputs)
    reRankCriterion.fullEvaluate(outputs, labels)
  }

  private def recommendItems(
    drModel: DeepRetrieval,
    sequenceIds: Array[Int],
    embedSize: Int,
    topk: Int,
    beamSize: Int,
    consumedItems: Set[Int]
  ): Array[Int] = {
    val candidateItems = searchCandidate(
      sequenceIds,
      drModel.layerModel,
      beamSize,
      drModel.pathItemsMapping
    ).toArray.filterNot(consumedItems.contains)

    val reRankScores = drModel.reRankModel.inference(
      candidateItems,
      sequenceIds,
      drModel.reRankWeights,
      drModel.reRankBias,
      embedSize
    )

    candidateItems
      .zip(reRankScores)
      .sortBy(_._2)(Ordering[Double].reverse)
      .take(topk)
      .map(_._1)
  }
}
