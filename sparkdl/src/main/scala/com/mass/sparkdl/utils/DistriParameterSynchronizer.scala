package com.mass.sparkdl.utils

import com.mass.sparkdl.tensor.Tensor

trait DistriParameterSynchronizer[T] {
  def init(name: String, globalSize: Int, priority: Int = 1, weights: Tensor[T],
    grads: Tensor[T]): Unit

  def put(name: String): Unit

  def get(name: String): (Tensor[T], Tensor[T])

  def clear(): Unit
}
