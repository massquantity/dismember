package com.mass.sparkdl.nn

import scala.concurrent.Future
import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.TensorModule
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}
import com.mass.sparkdl.utils.Engine

class FM[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  @transient private var results: Array[Future[Unit]] = _

  private var buffer: Array[T] = _
  // println("thread pool size: " + Engine.model.getPoolSize)

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
    if (results == null || results.length != batchSize) {
      results = new Array[Future[Unit]](batchSize)
    }
    var t = 0
    while (t < batchSize) {
      val _t = t
      results(_t) = Engine.model.invoke( () => {
        updateOutputOne(inData, outData, bufferData, _t, stride, featSize, embedSize)
      })
      t += 1
    }
    Engine.model.sync(results)
    output
  }

  def updateOutputOne(inData: Array[T], outData: Array[T], bufferData: Array[T],
      batchOffset: Int, stride: Int, featSize: Int, embedSize: Int): Unit = {
    val offset = batchOffset * stride
    val bufferOffset = batchOffset * embedSize
    var i = 0
    while (i < featSize) {
      val inOffset = offset + i * embedSize
      ev.vAdd(embedSize, bufferData, bufferOffset, inData, inOffset, bufferData, bufferOffset)
      i += 1
    }
    val sumSquare = ev.dot(embedSize, bufferData, bufferOffset, 1, bufferData, bufferOffset, 1)
    val squareSum = ev.dot(featSize * embedSize, inData, offset, 1, inData, offset, 1)
    outData(batchOffset) = ev.divide(ev.minus(sumSquare, squareSum), ev.fromType(2.0))
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).zero()
    val batchSize = input.size(0)
    val featSize = input.size(1)
    val embedSize = input.size(2)
    if (results != null || results.length != batchSize) {
      results = new Array[Future[Unit]](batchSize)
    }
    val stride = input.stride(0)
    val inData = input.storage().array()
    val gradInData = gradInput.storage().array()
    val gradOutData = gradOutput.storage().array()
    val bufferData = buffer
    var t = 0
    while (t < batchSize) {
      val _t = t
      results(_t) = Engine.model.invoke( () => {
        // updateGradInputOne(input.select(0, _t), gradInput.select(0, _t), _t)
        updateGradInputOne(inData, gradInData, gradOutData, bufferData, _t, stride, featSize, embedSize)
      })
      t += 1
    }
    Engine.model.sync(results)
    gradInput
  }

  def updateGradInputOne(inData: Array[T], gradInData: Array[T], gradOutData: Array[T],
      bufferData: Array[T], batchOffset: Int, stride: Int, featSize: Int, embedSize: Int): Unit = {
    val grad = gradOutData(batchOffset)
    val offset = batchOffset * stride
    val bufferOffset = batchOffset * embedSize
    var i = 0
    while (i < featSize) {
      val inOffset = offset + i * embedSize
      ev.vSub(embedSize, bufferData, bufferOffset, inData, inOffset, gradInData, inOffset)
      i += 1
    }
    ev.scal(featSize * embedSize, grad, gradInData, offset, 1)
  }

  override def clearState(): this.type = {
    super.clearState()
    buffer = Array.empty[T]
    results = null
    this
  }
}

object FM {
  def apply[@specialized(Float, Double) T: ClassTag]()(
      implicit ev: TensorNumeric[T]): FM[T] = {
    new FM[T]()
  }
}
