package com.mass.sparkdl.nn

import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.TensorModule
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}

class Linear[T: ClassTag](
    val inputSize: Int,
    val outputSize: Int,
    val withBias: Boolean = true)(implicit ev: TensorNumeric[T])
    extends TensorModule[T] {

  val weight: Tensor[T] = Tensor[T](outputSize, inputSize).randn(0.0, 0.05)
  val bias: Tensor[T] = if (withBias) Tensor[T](outputSize).zero() else null
  val addBuffer: Tensor[T] = Tensor[T]()

  val gradWeight: Tensor[T] = Tensor[T]()
  val gradBias: Tensor[T] = if (withBias) Tensor[T]() else null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2, "input dim must be 1D or 2D.")
    if (input.dim() == 1) {
      output.resize(Array(outputSize))
      if (withBias) output.copy(bias) else output.zero()
      output.addmv(ev.fromType[Int](1), weight, input)
    } else if (input.dim() == 2) {
      val batchSize = input.size(0)
      val nElement = output.nElement()
      output.resize(Array(batchSize, weight.size(0)))
      if (output.nElement() != nElement) output.zero()

      if (addBuffer.nElement() != batchSize) {
        addBuffer.resize(Array(batchSize)).fill(ev.one)
      }
      output.addmm(ev.zero, output, ev.one, input, weight.t())
      if (withBias) output.addr(ev.one, addBuffer, bias)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2, "input dim must be 1D or 2D.")
    val nElement = gradInput.nElement()
    gradInput.resizeAs(input)
    if (nElement != gradInput.nElement()) gradInput.zero()

    if (input.dim() == 1) {
      gradInput.addmv(ev.fromType[Int](0), ev.fromType[Int](1), weight.t(), gradOutput)
    } else if (input.dim() == 2) {
      gradInput.addmm(ev.fromType[Int](0), ev.fromType[Int](1), gradOutput, weight)
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    require(input.dim() == 1 || input.dim() == 2, "input dim must be 1D or 2D.")
    gradWeight.resize(outputSize, inputSize)
    if (withBias) gradBias.resize(outputSize)

    if (input.dim() == 1) {
      if (scaleW != 0) {
        gradWeight.addr(ev.fromType[Double](scaleW), gradOutput, input)
      }
      if (withBias && scaleB != 0) {
        gradBias.add(ev.fromType[Double](scaleB), gradOutput)
      }
    } else if (input.dim() == 2) {
      if (scaleW != 0) {
        gradWeight.addmm(ev.fromType[Double](scaleW), gradOutput.t(), input)
      }
      if (withBias && scaleB != 0) {
        gradBias.addmv(ev.fromType[Double](scaleB), gradOutput.t(), addBuffer)
      }
    }
  }

  override def clearState(): this.type = {
    super.clearState()
    addBuffer.set()
    this
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    if (null == bias) {
      (Array(this.weight), Array(this.gradWeight))
    } else {
      (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
    }
  }
}

object Linear {
  def apply[@specialized(Float, Double) T: ClassTag](
      inputSize: Int,
      outputSIze: Int,
      withBias: Boolean = true)(implicit ev: TensorNumeric[T]): Linear[T] = {
    new Linear[T](inputSize, outputSIze, withBias)
  }
}
