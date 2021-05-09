package com.mass.tdm.dataset

trait TDMSample extends Serializable {

  val sequence: Array[Int]

  val target: Int = -1

  val labels: Array[Int] = null
}

case class TDMTrainSample(sequence: Array[Int], override val target: Int) extends TDMSample {

  override def toString: String = {
    s"seq: ${sequence.mkString("Array(", ", ", ")")} | " +
      s"target: $target"
  }
}

case class TDMEvalSample(sequence: Array[Int], override val labels: Array[Int]) extends TDMSample {
  // override val target: Int = labels.head

  override def toString: String = {
    s"seq: ${sequence.mkString("Array(", ", ", ")")} | " +
      s"target: $target | " +
      s"labels: ${if (labels == null) "null" else labels.mkString("Array(", ", ", ")")}"
  }
}

/*
case class TDMSample[@specialized(Int, Long) T](sequence: Array[T], target: T) extends Serializable {
  override def toString: String = {
    s"seq: ${sequence.mkString(" ")} | " +
    s"target: $target"
  }
}

case class TDMEvalSample[@specialized(Int, Long) T](override val sequence: Array[T], labels: Array[T])
    extends TDMSample(sequence, labels.head) {
  override def toString: String = {
    s"seq: ${sequence.mkString(" ")} | " +
    s"target: $target | " +
    s"labels: ${if (labels == null) "null" else labels.mkString("Array(", ", ", ")")}"
  }
}
*/
// case class TDMSample[T](sequence: Array[Int], feature: Array[T], target: Int, label: Float)
