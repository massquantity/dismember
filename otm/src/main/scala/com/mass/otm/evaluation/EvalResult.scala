package com.mass.otm.evaluation

case class EvalResult(precision: Double, recall: Double, ndcg: Double) {

  def ++(other: EvalResult): EvalResult = {
    EvalResult(
      this.precision + other.precision,
      this.recall + other.recall,
      this.ndcg + other.ndcg
    )
  }

  def :/(size: Int): EvalResult = {
    EvalResult(
      precision / size,
      recall / size,
      ndcg / size
    )
  }

  override def toString: String = {
    f"{precision: $precision%.6f, recall: $recall%.6f, ndcg: $ndcg%.6f}"
  }
}
