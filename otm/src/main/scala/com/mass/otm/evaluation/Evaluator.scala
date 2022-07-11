package com.mass.otm.evaluation

import java.util.concurrent.Executors

import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.duration.Duration

import com.mass.otm.model.CandidateSearcher
import com.mass.otm.DeepModel
import com.mass.otm.dataset.LocalDataSet
import com.mass.otm.evaluation.Metrics.computeMetrics
import com.mass.otm.tree.OTMTree
import com.mass.sparkdl.nn.BCECriterionWithLogits
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble
import com.mass.sparkdl.Criterion
import com.mass.sparkdl.utils.Engine

object Evaluator extends Serializable with CandidateSearcher {

  val threadNum = Engine.coreNumber()

  implicit val criterions = (1 to threadNum).map(_ => BCECriterionWithLogits[Double](sizeAverage = false))

  implicit val context: ExecutionContext = ExecutionContext.fromExecutorService(
    Executors.newFixedThreadPool(threadNum)
  )
  // import scala.concurrent.ExecutionContext.Implicits.global

  def evaluate(
    models: IndexedSeq[DeepModel[Double]],
    dataset: LocalDataSet,
    tree: OTMTree,
    topk: Int,
    totalEvalBatchSize: Int,
    beamSize: Int,
    seqLen: Int,
    useMask: Boolean,
  ): (Double, EvalResult) = {
    val batchSize = math.max(1, totalEvalBatchSize / (beamSize * 2))
    val (totalLoss, totalMetrics) = dataset.evalData.sliding(batchSize, batchSize).toSeq.flatMap { batchData =>
      val threadDataSize = math.ceil(batchData.length.toDouble / threadNum).toInt
      batchData.sliding(threadDataSize, threadDataSize).toSeq.zipWithIndex.map { case (threadData, i) =>
        Future {
          val batchNodes = batchBeamSearch(threadData, models(i), tree, beamSize, seqLen, useMask)
          val (batchPreds, batchLabels, batchMetrics) = threadData.zip(batchNodes).map { case (data, nodes) =>
            val consumedItems = dataset.userConsumed(data.user).toSet
            val leafNodes = nodes
              .filterNot(n => consumedItems.contains(n.id))
              .filter(n => dataset.allNodes.contains(n.id))
              .sortBy(_.score)(Ordering[Double].reverse)
              .take(topk)
            val preds = leafNodes.map(_.score)
            val labels = leafNodes.map { n => data.targetItems.find(_ == n.id) match {
              case Some(_) => 1.0
              case None => 0.0
            }}
            val metrics = computeMetrics(leafNodes, data.targetItems)
            (preds, labels, metrics)
          }.unzip3
          val batchSumLoss = computeLoss(batchPreds, batchLabels, i)
          val batchSumMetrics = batchMetrics.reduce(_ ++ _)
          (batchSumLoss, batchSumMetrics)
        }
      }.map(future => Await.result(future, Duration.Inf))
    }.reduce((a, b) => (a._1 + b._1, a._2 ++ b._2))
    (totalLoss / dataset.evalSize, totalMetrics :/ dataset.evalSize)
  }

  def computeLoss(
    batchPreds: Seq[Seq[Double]],
    batchLabels: Seq[Seq[Double]],
    rank: Int
  )(implicit criterions: IndexedSeq[Criterion[Double]]): Double = {
    val preds = batchPreds.flatten.toArray
    val labels = batchLabels.flatten.toArray
    val predTensor = Tensor(preds, Array(preds.length, 1))
    val labelTensor = Tensor(labels, Array(labels.length, 1))
    criterions(rank).forward(predTensor, labelTensor)
  }
}
