package com.mass.sparkdl.nn

import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.AbstractModule
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}
import com.mass.sparkdl.utils.{T, Table}

class Mask[T: ClassTag](useScale: Boolean = false, factor: Int = -1)(
    implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  private val scaleFactor = if (useScale) 1.0 / math.sqrt(factor) else 1.0
  private val maskValue = Float.MinValue

  gradInput = T(Tensor[T](), Tensor[Int]())

  override def updateOutput(input: Table): Tensor[T] = {
    output.resizeAs(input(0)).copy(input(0))
    val outputData = output.storage().array()
    if (useScale) {
      ev.scal(outputData.length, ev.fromType(scaleFactor), outputData, 0, 1)
    }

    if (!input[Tensor[T]](1).isEmpty) {
      val num = input[Tensor[Int]](1).nElement()
      val maskData = input[Tensor[Int]](1).storage().array()
      var i = 0
      while (i < num) {
        val index = maskData(i)
        outputData(index) = ev.fromType(maskValue)
        i += 1
      }
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    gradInput[Tensor[T]](0).resizeAs(gradOutput).copy(gradOutput)
    val gradInputData = gradInput[Tensor[T]](0).storage().array()
    if (useScale) {
      ev.scal(gradInputData.length, ev.fromType(scaleFactor), gradInputData, 0, 1)
    }

    if (!input[Tensor[Int]](1).isEmpty) {
      val num = input[Tensor[Int]](1).nElement()
      val maskData = input[Tensor[Int]](1).storage().array()
      var i = 0
      while (i < num) {
        val index = maskData(i)
        gradInputData(index) = ev.fromType(0.0)
        i += 1
      }
    }

    gradInput[Tensor[Int]](1).resizeAs(input(1)).zero()
    gradInput
  }

  override def clearState(): this.type = {
    super.clearState()
    this
  }
}

object Mask {
  def apply[@specialized(Float, Double) T: ClassTag](
      useScale: Boolean = false, factor: Int = -1)(
      implicit ev: TensorNumeric[T]): Mask[T] = {
    new Mask[T](useScale, factor)
  }
}
