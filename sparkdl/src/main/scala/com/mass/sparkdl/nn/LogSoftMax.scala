package com.mass.sparkdl.nn

import scala.concurrent.Future
import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}
import com.mass.sparkdl.utils.Engine

class LogSoftMax[T: ClassTag](implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  @transient private var results: Array[Future[Unit]] = _
  private val ones: Tensor[T] = Tensor[T]()
  private val buffer: Tensor[T] = Tensor[T]()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2, "input dim must be 1D or 2D")
    output.resizeAs(input).copy(input)
    if (input.dim() == 1) {
      updateOutputOne(input, output)
    } else {
      val batchSize = input.size(0)
      if (results == null || results.length != batchSize) {
        results = new Array[Future[Unit]](batchSize)
      }
      var t = 0
      while (t < batchSize) {
        val _t = t
        results(_t) = Engine.model.invoke( () => {
          updateOutputOne(input.select(0, _t), output.select(0, _t))
        })
        t += 1
      }
      Engine.model.sync(results)
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
      if (results == null || results.length != batchSize) {
        results = new Array[Future[Unit]](batchSize)
      }
      var t = 0
      while (t < batchSize) {
        val _t = t
        results(_t) = Engine.model.invoke( () => {
          updateGradInputOne(output.select(0, _t), gradInput.select(0, _t))
        })
        t += 1
      }
      Engine.model.sync(results)
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
    results = null
    this
  }
}

object LogSoftMax {
  def apply[@specialized(Float, Double) T: ClassTag]()(
      implicit ev: TensorNumeric[T]): LogSoftMax[T] = {
    new LogSoftMax[T]()
  }
}
