package com.mass.sparkdl.nn

import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.TensorModule
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}

class SoftMax[T: ClassTag](implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  private var buffer: Array[T] = _
  private var ones: Array[T] = _

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
    val dim = input.size().last
    val n = input.size().product / dim
    if (ones == null) ones = Array.fill[T](dim)(ev.one)

    val inputData =
      if (input.isContiguous) {
        input.storage().array()
      } else {
        input.contiguous().storage().array()
      }
    val outputData = output.storage().array()

    var i = 0
    while (i < n) {
      val offset = i * dim

      var inputMax = inputData(offset)
      var d = 1
      while (d < dim) {
        inputMax = ev.max(inputMax, inputData(offset + d))
        d += 1
      }

      buffer = Array.fill[T](dim)(inputMax)
      ev.vSub(dim, inputData, offset, buffer, 0, outputData, offset)
      ev.vExp(dim, outputData, offset, outputData, offset)
      val sum = ev.dot(dim, outputData, offset, 1, ones, 0, 1)
      ev.scal(dim, ev.inv(sum), outputData, offset, 1)
      i += 1
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)
    val dim = input.size().last
    val n = input.size().product / dim

    val gradInputData = gradInput.storage().array()
    val outputData = output.storage().array()
    val gradOutputData = gradOutput.storage().array()

    var i = 0
    while (i < n) {
      val offset = i * dim

      val sum = ev.dot(dim, gradOutputData, offset, 1, outputData, offset, 1)
      buffer = Array.fill[T](dim)(sum)
      ev.vSub(dim, gradOutputData, offset, buffer, 0, gradInputData, offset)
      ev.vMul(dim, gradInputData, offset, outputData, offset, gradInputData, offset)
      i += 1
    }
    gradInput
  }
}

object SoftMax {
  def apply[@specialized(Float, Double) T: ClassTag]()(
      implicit ev: TensorNumeric[T]): SoftMax[T] = {
    new SoftMax[T]()
  }
}
