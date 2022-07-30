package com.mass.dr.evaluation

import com.mass.dr.dataset.LocalDataSet
import com.mass.dr.evaluation.Metrics.computeMetrics
import com.mass.dr.model.{CandidateSearcher, LayerModel, MappingOp, RerankModel}
import com.mass.dr.dataset.MiniBatch.{LayerTransformedBatch, RerankTransformedBatch}
import com.mass.dr.loss.CrossEntropyLayer
import com.mass.dr.Path
import com.mass.scalann.nn.SampledSoftmaxLoss
import com.mass.scalann.tensor.Tensor
import com.mass.scalann.utils.Engine

object Evaluator extends Serializable with CandidateSearcher {

  def evaluate(
    dataset: LocalDataSet,
    layerModel: LayerModel,
    reRankModel: RerankModel,
    layerCriterion: CrossEntropyLayer,
    reRankCriterion: SampledSoftmaxLoss[Double],
    topk: Int,
    beamSize: Int,
    itemPathMapping: Map[Int, Seq[Path]]
  ): EvalResult = {
    val subModelNum = Engine.coreNumber()
    val pathItemMapping = MappingOp.pathToItems(itemPathMapping)
    val userConsumed = dataset.getUserConsumed
    val allData = dataset.getEvalData
    val miniBatchIter = dataset.iteratorMiniBatch(train = false)
    val evalResults = miniBatchIter.map { batch =>
      val offset = batch.getOffset
      val length = batch.getLength
      val layerMiniBatch = batch.transformLayerData(allData, offset, length, itemPathMapping)
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
            val recItems = recommendItems(
              layerModel,
              reRankModel,
              data.sequence,
              topk,
              beamSize,
              consumedItems,
              pathItemMapping
            )
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
    layerModel: LayerModel,
    reRankModel: RerankModel,
    sequenceIds: Seq[Int],
    topk: Int,
    beamSize: Int,
    consumedItems: Set[Int],
    pathItemsMapping: Map[Path, Seq[Int]]
  ): Seq[Int] = {
    val candidateItems = searchCandidate(
      sequenceIds.toIndexedSeq,
      layerModel,
      beamSize,
      pathItemsMapping
    ).filterNot(consumedItems.contains)

    val reRankScores = reRankModel.inference(candidateItems, sequenceIds)
    candidateItems
      .zip(reRankScores)
      .sortBy(_._2)(Ordering[Double].reverse)
      .take(topk)
      .map(_._1)
  }
}
