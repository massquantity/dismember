package com.mass.dr.dataset

trait DRSample extends Serializable {

  val sequence: Array[Int]

  val target: Int

  val labels: Array[Int]

  val user: Int
}

case class DRTrainSample(
  override val sequence: Array[Int],
  override val target: Int) extends DRSample {

  override val labels: Array[Int] = null

  override val user: Int = -1

  override def toString: String = {
    s"seq: ${sequence.mkString("Array(", ", ", ")")} | " +
      s"target: $target"
  }
}

case class DREvalSample(
  override val sequence: Array[Int],
  override val labels: Array[Int],
  override val user: Int) extends DRSample {

  override val target: Int = labels.head

  override def toString: String = {
    s"seq: ${sequence.mkString("Array(", ", ", ")")} | " +
      s"target: $target | " +
      s"labels: ${if (labels == null) "null" else labels.mkString("Array(", ", ", ")")}"
  }
}
