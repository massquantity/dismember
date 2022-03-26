package com.mass.otm.dataset

sealed trait OTMSample extends Product with Serializable {

  val sequence: Array[Int]

  val target: Int

  val labels: Array[Int]

  val user: Int
}

object OTMSample {

  case class OTMTrainSample(
    override val sequence: Array[Int],
    override val target: Int
  ) extends OTMSample {

    override val labels: Array[Int] = null

    override val user: Int = -1

    override def toString: String = {
      s"seq: ${sequence.mkString("Array(", ", ", ")")} | " +
        s"target: $target"
    }
  }

  case class OTMEvalSample(
    override val sequence: Array[Int],
    override val labels: Array[Int],
    override val user: Int
  ) extends OTMSample {

    override val target: Int = labels.head

    override def toString: String = {
      s"seq: ${sequence.mkString("Array(", ", ", ")")} | " +
        s"target: $target | " +
        s"labels: ${if (labels == null) "null" else labels.mkString("Array(", ", ", ")")}"
    }
  }
}
