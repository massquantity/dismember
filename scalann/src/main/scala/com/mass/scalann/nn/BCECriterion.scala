package com.mass.scalann.nn

import scala.reflect.ClassTag

import com.mass.scalann.nn.abstractnn.AbstractCriterion
import com.mass.scalann.tensor.{Tensor, TensorNumeric}

class BCECriterion[@specialized(Float, Double) T: ClassTag](sizeAverage: Boolean = true)(
    implicit ev: TensorNumeric[T]) extends AbstractCriterion[T] {

  private val eps = 1e-12
  val buffer: Tensor[T] = Tensor[T]()

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require(input.size().sameElements(target.size()),
      s"input size should be equal to target size, but got input size: ${input.size().toSeq}," +
        s" target size: ${target.size().toSeq}")

    var sum = 0.0
    // y * log(x)
    buffer.resizeAs(input).copy(input).add(ev.fromType(eps)).log()
    sum += ev.toType[Double](buffer.dot(target))
    // (1 - y) * log(1 - x) = log(1 - x) - y * log(1 - x)
    buffer.fill(ev.one).sub(input).add(ev.fromType(eps)).log()
    sum += ev.toType[Double](buffer.sum())
    sum -= ev.toType[Double](buffer.dot(target))

    if (sizeAverage) sum /= input.nElement()
    output = ev.fromType[Double](-sum)
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    val nElement = input.nElement()
    val norm = if (sizeAverage) 1.0 / nElement else 1.0
    gradInput.resizeAs(input)
    buffer.pow(input, ev.fromType(2)).sub(input).sub(ev.fromType(eps))
    gradInput.copy(target).sub(input).cdiv(buffer).mul(ev.fromType(norm))
    gradInput
  }
}

object BCECriterion {
  def apply[@specialized(Float, Double) T: ClassTag](sizeAverage: Boolean = true)(
      implicit ev: TensorNumeric[T]): BCECriterion[T] = {
    new BCECriterion[T](sizeAverage)
  }
}
