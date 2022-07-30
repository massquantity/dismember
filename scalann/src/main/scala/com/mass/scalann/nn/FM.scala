package com.mass.scalann.nn

import scala.reflect.ClassTag

import com.mass.scalann.nn.abstractnn.TensorModule
import com.mass.scalann.tensor.{Tensor, TensorNumeric}

class FM[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  private var buffer: Array[T] = _

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3, "FM input must be 3D")
    val batchSize = input.size(0)
    val featSize = input.size(1)
    val embedSize = input.size(2)
    buffer = new Array[T](batchSize * embedSize)
    output.resize(Array(batchSize, 1))
    val inData = input.storage().array()
    val outData = output.storage().array()
    val bufferData = buffer
    val stride = input.stride(0)

    var t = 0
    while (t < batchSize) {
      val offset = t * stride
      val bufferOffset = t * embedSize
      var i = 0
      while (i < featSize) {
        val inOffset = offset + i * embedSize
        ev.vAdd(embedSize, bufferData, bufferOffset, inData, inOffset, bufferData, bufferOffset)
        i += 1
      }
      val sumSquare = ev.dot(embedSize, bufferData, bufferOffset, 1, bufferData, bufferOffset, 1)
      val squareSum = ev.dot(featSize * embedSize, inData, offset, 1, inData, offset, 1)
      outData(t) = ev.divide(ev.minus(sumSquare, squareSum), ev.fromType(2.0))
      t += 1
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).zero()
    val batchSize = input.size(0)
    val featSize = input.size(1)
    val embedSize = input.size(2)

    val stride = input.stride(0)
    val inData = input.storage().array()
    val gradInData = gradInput.storage().array()
    val gradOutData = gradOutput.storage().array()
    val bufferData = buffer

    var t = 0
    while (t < batchSize) {
      val grad = gradOutData(t)
      val offset = t * stride
      val bufferOffset = t * embedSize
      var i = 0
      while (i < featSize) {
        val inOffset = offset + i * embedSize
        ev.vSub(embedSize, bufferData, bufferOffset, inData, inOffset, gradInData, inOffset)
        i += 1
      }
      ev.scal(featSize * embedSize, grad, gradInData, offset, 1)

      t += 1
    }
    gradInput
  }

  override def clearState(): this.type = {
    super.clearState()
    buffer = Array.empty[T]
    this
  }
}

object FM {
  def apply[@specialized(Float, Double) T: ClassTag]()(
      implicit ev: TensorNumeric[T]): FM[T] = {
    new FM[T]()
  }
}
