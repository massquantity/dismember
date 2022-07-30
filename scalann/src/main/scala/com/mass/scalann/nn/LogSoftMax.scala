package com.mass.scalann.nn

import scala.reflect.ClassTag

import com.mass.scalann.nn.abstractnn.TensorModule
import com.mass.scalann.tensor.{Tensor, TensorNumeric}

class LogSoftMax[T: ClassTag](implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  private val ones: Tensor[T] = Tensor[T]()
  private val buffer: Tensor[T] = Tensor[T]()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2, "input dim must be 1D or 2D")
    output.resizeAs(input).copy(input)
    if (input.dim() == 1) {
      updateOutputOne(input, output)
    } else {
      val batchSize = input.size(0)
      var t = 0
      while (t < batchSize) {
        updateOutputOne(input.select(0, t), output.select(0, t))
        t += 1
      }
    }
    output
  }

  private def updateOutputOne(in: Tensor[T], out: Tensor[T]): Unit = {
    if (ones.nElement() < in.nElement()) ones.resizeAs(in).fill(ev.one)
    if (buffer.nElement() < out.nElement()) buffer.resizeAs(out)

    // use exp(in - maxInput) to avoid Infinity error
    val maxInput = in.max()
    buffer.fill(ev.negative(maxInput))
    buffer.add(in)
    buffer.exp()
    val logSum = ev.plus(maxInput, ev.log(buffer.dot(ones)))
    out.add(ev.negative(logSum))
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(gradOutput.dim() == input.dim())
    gradInput.resizeAs(input).copy(gradOutput)
    if (output.dim() == 1) {
      updateGradInputOne(output, gradInput)
    } else {
      val batchSize = output.size(0)
      var t = 0
      while (t < batchSize) {
        updateGradInputOne(output.select(0, t), gradInput.select(0, t))
        t += 1
      }
    }
    gradInput
  }

  private def updateGradInputOne(out: Tensor[T], gradOut: Tensor[T]): Unit = {
    buffer.exp(out)
    val outSum = gradOut.dot(ones)
    gradOut.add(ev.negative(outSum), buffer)
  }

  override def clearState() : this.type = {
    super.clearState()
    ones.set()
    buffer.set()
    this
  }
}

object LogSoftMax {
  def apply[@specialized(Float, Double) T: ClassTag]()(
    implicit ev: TensorNumeric[T]): LogSoftMax[T] = {
    new LogSoftMax[T]()
  }
}
