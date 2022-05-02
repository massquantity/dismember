package com.mass.otm.evaluation

import com.mass.otm.model.Recommender
import com.mass.otm.DeepModel
import com.mass.otm.dataset.LocalDataSet
import com.mass.otm.evaluation.Metrics.computeMetrics
import com.mass.otm.tree.OTMTree
import com.mass.otm.tree.OTMTree.Node
import com.mass.sparkdl.nn.BCECriterionWithLogits
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble
import com.mass.sparkdl.Criterion

object Evaluator extends Serializable with Recommender {

  implicit val criterion = BCECriterionWithLogits[Double](sizeAverage = false)

  def evaluate(
    model: DeepModel[Double],
    dataset: LocalDataSet,
    tree: OTMTree,
    topk: Int,
    totalEvalBatchSize: Int,
    beamSize: Int,
    seqLen: Int,
    useMask: Boolean,
  ): (Double, EvalResult) = {
    val validLeaves = dataset.idItemMapping.keySet
    val batchSize = math.max(1, totalEvalBatchSize / (beamSize * 2))
    // val numBatch = math.ceil(dataset.evalSize.toDouble / batchSize).toInt
    val (totalLoss, totalMetrics) = dataset.evalData.sliding(batchSize, batchSize).map { batchData =>
      val batchNodes: Seq[Seq[Node]] = batchBeamSearch(batchData, model, tree, beamSize, seqLen, useMask)
      val (batchPreds, batchLabels, batchMetrics) = batchData.zip(batchNodes).map { case (data, nodes) =>
        val consumedItems = dataset.userConsumed(data.user).toSet
        val leafNodes = nodes
          .filterNot(n => consumedItems.contains(n.id))
          .filter(n => validLeaves.contains(n.id))
          .sortBy(_.pred)(Ordering[Double].reverse)
          .take(topk)
        val preds = leafNodes.map(_.pred)
        val labels = leafNodes.map(i => if (i.id == data.target) 1.0 else 0.0)
        val metrics = computeMetrics(leafNodes, data.labels)
        (preds, labels, metrics)
      }.unzip3
      val batchSumLoss = computeLoss(batchPreds, batchLabels)
      val batchSumMetrics = batchMetrics.reduce(_ ++ _)
      (batchSumLoss, batchSumMetrics)
    }.reduce((a, b) => (a._1 + b._1, a._2 ++ b._2))
    (totalLoss / dataset.evalSize, totalMetrics :/ dataset.evalSize)
  }

  def computeLoss(
    batchPreds: Seq[Seq[Double]],
    batchLabels: Seq[Seq[Double]]
  )(implicit criterion: Criterion[Double]): Double = {
    val preds = batchPreds.flatten.toArray
    val labels = batchLabels.flatten.toArray
    val predTensor = Tensor(preds, Array(preds.length, 1))
    val labelTensor = Tensor(labels, Array(labels.length, 1))
    criterion.forward(predTensor, labelTensor)
  }
}
