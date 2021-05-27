package com.mass.sparkdl.nn

import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.TensorModule
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}

class SoftMax[T: ClassTag](implicit ev: TensorNumeric[T]) extends TensorModule[T]{

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
    val dim = input.size().last
    val n = input.size().product / dim

    val inputData = {
      if (input.isContiguous) {
        input.storage().array()
      } else {
        input.contiguous().storage().array()
      }
    }
    val outputData = output.storage().array()

    var i = 0
    while (i < n) {
      val offset = i * dim

      var inputMax = inputData(0)
      var d = 1
      while (d < dim) {
        inputMax = ev.max(inputMax, inputData(offset + d))
        d += 1
      }

      var sum = ev.zero
      d = 0
      while (d < dim) {
        val index = offset + d
        val z = ev.exp(ev.minus(inputData(index), inputMax))
        outputData(index) = z
        sum = ev.plus(sum, z)
        d += 1
      }

      d = 0
      while (d < dim) {
        val index = offset + d
        outputData(index) = ev.divide(outputData(index), sum)
        d += 1
      }

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

      var sum = ev.zero
      var d = 0
      while (d < dim) {
        val index = offset + d
        sum = ev.plus(sum,
          ev.times(gradOutputData(index), outputData(index))
        )
        d += 1
      }

      d = 0
      while (d < dim) {
        val index = offset + d
        gradInputData(index) = ev.times(outputData(index),
          ev.minus(gradOutputData(index), sum))
        d += 1
      }

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
