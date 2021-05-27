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
    if (input.dim() == 1) {
      output.resize(Array(outputSize))
      if (withBias) output.copy(bias) else output.zero()
      output.addmv(ev.fromType[Int](1), weight, input)
    } else {
      val batchSize = {
        if (input.dim() == 2) {
          input.size(0)
        } else {
          input.size().init.product
        }
      }
      val _input = {
        if (input.dim() == 2) {
          input
        } else {
          input.view(Array(batchSize, inputSize))
        }
      }

      output.resize(Array(batchSize, outputSize))
      output.addmm(ev.zero, ev.one, _input, weight.t())

      if (addBuffer.nElement() != batchSize) {
        addBuffer.resize(Array(batchSize)).fill(ev.one)
      }
      if (withBias) {
        output.addr(ev.one, addBuffer, bias)
      }
      if (input.dim() > 2) {
        val _outputSize = input.size().init ++ Array(outputSize)
        output = output.view(_outputSize)
      }
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (input.dim() <= 2) {
      gradInput.resizeAs(input)
    } else {
      val gradInputSize = Array(input.size().init.product, inputSize)
      gradInput.resize(gradInputSize)
    }
    val _gradOutput = {
      if (input.dim() <= 2) {
        gradOutput
      } else {
        val gradOutputSize = Array(input.size().init.product, outputSize)
        gradOutput.view(gradOutputSize)
      }
    }

    if (input.dim() == 1) {
      gradInput.addmv(ev.zero, ev.one, weight.t(), _gradOutput)
    } else {
      gradInput.addmm(ev.zero, ev.one, _gradOutput, weight)
    }

    if (input.dim() > 2) {
      gradInput = gradInput.view(input.size())
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    gradWeight.resize(outputSize, inputSize)
    if (withBias) gradBias.resize(outputSize)

    val (_input, _gradOutput) = {
      if (input.dim() <= 2) {
        (input, gradOutput)
      } else {
        val flatSize = input.size().init.product
        (input.view(Array(flatSize, inputSize)), gradOutput.view(Array(flatSize, outputSize)))
      }
    }

    if (input.dim() == 1) {
      if (scaleW != 0) {
        gradWeight.addr(ev.fromType[Double](scaleW), gradOutput, input)
      }
      if (withBias && scaleB != 0) {
        gradBias.add(ev.fromType[Double](scaleB), gradOutput)
      }
    } else {
      if (scaleW != 0) {
        gradWeight.addmm(ev.fromType[Double](scaleW), _gradOutput.t(), _input)
      }
      if (withBias && scaleB != 0) {
        gradBias.addmv(ev.fromType[Double](scaleB), _gradOutput.t(), addBuffer)
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
