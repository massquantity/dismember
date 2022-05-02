package com.mass.otm.dataset

sealed trait OTMSample extends Product with Serializable {

  val sequence: Seq[Int]

  val target: Int

  val labels: Seq[Int]

  val user: Int
}

object OTMSample {

  case class OTMTrainSample(
    override val sequence: Seq[Int],
    override val target: Int
  ) extends OTMSample {

    override val labels: Seq[Int] = null

    override val user: Int = -1

    override def toString: String = {
      s"seq: ${sequence.mkString("Seq(", ", ", ")")} | " +
        s"target: $target"
    }
  }

  case class OTMEvalSample(
    override val sequence: Seq[Int],
    override val labels: Seq[Int],
    override val user: Int
  ) extends OTMSample {

    override val target: Int = labels.head

    override def toString: String = {
      s"seq: ${sequence.mkString("Seq(", ", ", ")")} | " +
        s"target: $target | " +
        s"labels: ${if (labels == null) "null" else labels.mkString("Seq(", ", ", ")")}"
    }
  }
}
