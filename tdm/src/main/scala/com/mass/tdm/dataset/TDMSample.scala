package com.mass.tdm.dataset

trait TDMSample extends Serializable {

  val sequence: Array[Int]

  val target: Int

  val labels: Array[Int]
}

case class TDMTrainSample(sequence: Array[Int], override val target: Int) extends TDMSample {
  override val labels: Array[Int] = null

  override def toString: String = {
    s"seq: ${sequence.mkString("Array(", ", ", ")")} | " +
      s"target: $target"
  }
}

case class TDMEvalSample(sequence: Array[Int], override val labels: Array[Int]) extends TDMSample {
  override val target: Int = labels.head

  override def toString: String = {
    s"seq: ${sequence.mkString("Array(", ", ", ")")} | " +
      s"target: $target | " +
      s"labels: ${if (labels == null) "null" else labels.mkString("Array(", ", ", ")")}"
  }
}
