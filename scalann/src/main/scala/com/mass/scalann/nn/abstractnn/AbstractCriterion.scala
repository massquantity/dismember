package com.mass.scalann.nn.abstractnn

import scala.reflect.ClassTag

import com.mass.scalann.tensor.{Tensor, TensorNumeric}
import org.apache.commons.lang3.SerializationUtils

abstract class AbstractCriterion[T: ClassTag](implicit ev: TensorNumeric[T]) extends Serializable {
  var output: T = ev.fromType[Int](0)
  var gradInput: Tensor[T] = Tensor[T]()

  def forward(input: Tensor[T], target: Tensor[T]): T = {
    updateOutput(input, target)
  }

  def backward(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    updateGradInput(input, target)
  }

  def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    this.output
  }

  def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T]

  def cloneCriterion(): AbstractCriterion[T] = {
    SerializationUtils.clone(this)
  }
}
