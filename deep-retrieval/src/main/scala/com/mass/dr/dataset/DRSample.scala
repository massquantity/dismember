package com.mass.dr.dataset

case class DRSample(sequence: Array[Int], target: Int) {

  override def toString: String = {
    s"seq: ${sequence.mkString("Array(", ", ", ")")} | " +
      s"target: $target"
  }
}
