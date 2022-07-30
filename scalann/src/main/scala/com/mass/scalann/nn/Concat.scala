package com.mass.scalann.nn

import scala.reflect.ClassTag

import com.mass.scalann.nn.abstractnn.AbstractModule
import com.mass.scalann.tensor.{Tensor, TensorNumeric}
import com.mass.scalann.utils.Table

class Concat[T: ClassTag](flatten: Boolean = false)(
    implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  override def updateOutput(input: Table): Tensor[T] = {
    val frontSize = input[Tensor[T]](0).size().init
    var lastDim = input[Tensor[T]](0).size().last
    var i = 1
    while (i < input.length) {
      val size = input[Tensor[T]](i).size()
      require(frontSize.sameElements(size.init), "size should be same except for last dim")
      lastDim += size.last
      i += 1
    }
    output.resize(frontSize ++ Array(lastDim))

    val outputSize = output.size().last
    val outputData = output.storage().array()
    var dimPosition = 0
    i = 0
    while (i < input.length) {
      val inputTensor = input[Tensor[T]](i)
      val inputData = inputTensor.storage().array()
      val inputSize = inputTensor.size().last
      val n = inputTensor.size().init.product
      var j = 0
      while (j < n) {
        val inputOffset = j * inputSize
        val outputOffset = j * outputSize + dimPosition
        System.arraycopy(inputData, inputOffset,
          outputData, outputOffset, inputSize)
        j += 1
      }
      dimPosition += inputSize
      i += 1
    }
    if (flatten) output.squeeze() else output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    val gradOutputSize = gradOutput.size().last
    val gradOutputData = gradOutput.storage().array()
    var dimPosition = 0
    var i = 0
    while (i < input.length) {
      val inputTensor = input[Tensor[T]](i)
      if (!gradInput.contains(i)) {
        gradInput(i) = Tensor[T]().resizeAs(inputTensor)
      } else {
        gradInput[Tensor[T]](i).resizeAs(inputTensor)
      }

      val gradInputData = gradInput[Tensor[T]](i).storage().array()
      val gradInputSize = gradInput[Tensor[T]](i).size().last
      val n = inputTensor.size().init.product
      var j = 0
      while (j < n) {
        val gradInputOffset = j * gradInputSize
        val gradOutputOffset = j * gradOutputSize + dimPosition
        System.arraycopy(gradOutputData, gradOutputOffset,
          gradInputData, gradInputOffset, gradInputSize)
        j += 1
      }
      dimPosition += gradInputSize
      i += 1
    }

    gradInput
  }

  override def clearState(): Concat.this.type = {
    super.clearState()
    gradInput.clear()
    this
  }
}

object Concat {
  def apply[@specialized(Float, Double) T: ClassTag](flatten: Boolean = false)(
      implicit ev: TensorNumeric[T]): Concat[T] = {
    new Concat[T](flatten)
  }
}
