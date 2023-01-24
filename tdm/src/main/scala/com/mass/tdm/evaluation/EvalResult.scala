package com.mass.tdm.evaluation

private[tdm] class EvalResult(
    private[evaluation] var loss: Double = 0.0,
    private[evaluation] var precision: Double = 0.0,
    private[evaluation] var recall: Double = 0.0,
    private[evaluation] var ndcg: Double = 0.0,
    private[evaluation] var count: Int = 0
) extends Serializable {

  def +(other: EvalResult): EvalResult = {
    this.loss += other.loss
    this.precision += other.precision
    this.recall += other.recall
    this.ndcg += other.ndcg
    this.count += other.count
    this
  }

  def +=(values: (Double, Double, Double)): Unit = {
    this.precision += values._1
    this.recall += values._2
    this.ndcg += values._3
  }

  override def toString: String = {
    f"{eval loss: ${loss / count}%.4f, " +
      f"precision: ${precision / count}%.6f, " +
      f"recall: ${recall / count}%.6f, " +
      f"ndcg: ${ndcg / count}%.6f}"
  }
}
