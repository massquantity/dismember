package com.mass.scalann.tensor

trait TensorFunc2[@specialized(Float, Double) T] {

  def apply(v1: Array[T], v2: Int): Unit

  override def toString: String = "<TensorFunction2>"
}

trait TensorFunc4[@specialized(Float, Double) T] {

  def apply(v1: Array[T], v2: Int, v3: Array[T], v4: Int): Unit

  override def toString: String = "<TensorFunction4>"
}
