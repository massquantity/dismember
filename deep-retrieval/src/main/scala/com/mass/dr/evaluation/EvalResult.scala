package com.mass.dr.evaluation

import java.text.DecimalFormat

case class EvalResult(
    layerLoss: Seq[Double],
    reRankLoss: Double,
    precision: Double,
    recall: Double,
    ndcg: Double,
    size: Int,
    count: Int = 1
) {
  def +(other: EvalResult): EvalResult = {
    EvalResult(
      this.layerLoss.lazyZip(other.layerLoss).map(_ + _),
      this.reRankLoss + other.reRankLoss,
      this.precision + other.precision,
      this.recall + other.recall,
      this.ndcg + other.ndcg,
      this.size + other.size,
      this.count + other.count
    )
  }

  def meanMetrics: EvalResult = {
    EvalResult(
      layerLoss.map(_ / count),
      reRankLoss / count,
      precision / size,
      recall / size,
      ndcg / size,
      size,
      count
    )
  }

  override def toString: String = {
    val stringRepr = new StringBuilder
    val formatter = new DecimalFormat("##.####")
    val layerLossStr = layerLoss.map(i => formatter.format(i / count))
    stringRepr ++= s"eval layer loss: ${layerLossStr.mkString("[",", ","]")}, "
    stringRepr ++= f"rerank loss: ${reRankLoss / count}%.4f\n"
    stringRepr ++= f"\t\tprecision: ${precision / size}%.6f, "
    stringRepr ++= f"recall: ${recall / size}%.6f, "
    stringRepr ++= f"ndcg: ${ndcg / size}%.6f"
    stringRepr.toString
  }
}
