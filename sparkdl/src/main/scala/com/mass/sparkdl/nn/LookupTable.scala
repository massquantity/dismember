package com.mass.sparkdl.nn

import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}

class LookupTable[T: ClassTag](val nIndex: Int, val nOutput: Int)(
    implicit ev: TensorNumeric[T]) extends TensorModule[T]{

  var weight: Tensor[T] = Tensor[T](nIndex, nOutput).randn(0.0, 0.05)
  var gradWeight: Tensor[T] = Tensor[T](nIndex, nOutput).zero()
  private var inputBuffer = Tensor[Int]()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
  //  println(input.storage().array().asInstanceOf[Array[Int]]
  //    .sorted(Ordering.by[Int, Int](-_)).slice(0, 10).mkString(" "))
    inputBuffer = input.contiguous().asInstanceOf[Tensor[Int]]
    if (input.dim() == 2) {
      inputBuffer = inputBuffer.view(inputBuffer.nElement())
    }

    val numEle = inputBuffer.nElement()
    val newSize = weight.size().clone()
    newSize(0) = numEle
    output.resize(newSize)
    try {
      var i = 0
      while (i < numEle) {
        output.select(0, i).copy(weight.select(0, inputBuffer(Array(i))))
        i += 1
      }
      if (input.dim() == 2) {
        output = output.view(input.size(0), input.size(1), weight.size(1))
      }
    } catch {
      case e: IllegalArgumentException =>
        throw new IllegalArgumentException(
          s"""LookupTable updateOutput get exception: "${e.getMessage}"\n""" +
          s"please ensure elements of your input will not exceed $nIndex")

      case e: Exception =>
        throw e
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!gradInput.isSameSizeAs(input)) {
      gradInput.resizeAs(input).zero()
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    require(gradWeight.isContiguous, "LookupTable: gradWeight must be contiguous")
    inputBuffer = input.contiguous().asInstanceOf[Tensor[Int]]
    val inputData: Array[Int] = inputBuffer.storage().array()
    val inputOffset: Int = inputBuffer.storageOffset()
    val numEle: Int = inputBuffer.nElement()
    val _gradOutput = gradOutput.contiguous()
    val gradWeightData = gradWeight.storage().array()
    val gradWeightOffset = gradWeight.storageOffset()
    val gradOutputData = _gradOutput.storage().array()
    val gradOutputOffset = _gradOutput.storageOffset()
    val stride = gradWeight.stride(1)

    var i = 0
    while (i < numEle) {
      val index = inputData(i + inputOffset)
      ev.axpy(stride, ev.fromType(scaleW), gradOutputData, i * stride + gradOutputOffset, 1,
        gradWeightData, index * stride + gradWeightOffset, 1)
      i += 1
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight), Array(this.gradWeight))
  }

  override def clearState(): this.type = {
    super.clearState()
    inputBuffer.set()
    this
  }
}

object LookupTable {
  def apply[@specialized(Float, Double) T: ClassTag](nIndex: Int, nOutput: Int)(
      implicit ev: TensorNumeric[T]): LookupTable[T] = {
    new LookupTable[T](nIndex, nOutput)
  }
}
