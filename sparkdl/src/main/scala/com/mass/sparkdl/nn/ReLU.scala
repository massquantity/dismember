package com.mass.sparkdl.nn

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future
import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.TensorModule
import com.mass.sparkdl.tensor.{DoubleType, FloatType, Tensor, TensorNumeric}
import com.mass.sparkdl.utils.Engine

class ReLU[T: ClassTag](implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.isContiguous, "input must be contiguous")
    val threadNum = Engine.model.getPoolSize
    val taskSize = input.nElement() / threadNum
    var extraTaskSize = input.nElement() % threadNum
    var allocated = 0
    val tasks = new ArrayBuffer[(Int, Int)]()
    while (allocated < input.nElement()) {
      val end = {
        if (extraTaskSize > 0) {
          extraTaskSize -= 1
          allocated + taskSize + 1
        } else {
          allocated + taskSize
        }
      }
      tasks += ((allocated, math.min(input.nElement(), end)))
      allocated = end
    }
    val taskArray = tasks.toArray
    val results = new Array[Future[Unit]](taskArray.length)

    ev.getType match {
      case DoubleType =>
        output.asInstanceOf[Tensor[Double]].resizeAs(input.asInstanceOf[Tensor[Double]])
        val inputDouble = input.asInstanceOf[Tensor[Double]]
        val inputData = inputDouble.storage().array()
        val inputOffset = inputDouble.storageOffset()
        val outputDouble = output.asInstanceOf[Tensor[Double]]
        val outputData = outputDouble.storage().array()
        val outputOffset = outputDouble.storageOffset()

        var t = 0
        while (t < taskArray.length) {
          val _t = t
          results(_t) = Engine.model.invoke( () => {
            var i = taskArray(_t)._1
            while (i < taskArray(_t)._2) {
              outputData(outputOffset + i) = math.max(inputData(inputOffset + i), 0.0)
              i += 1
            }
          })
          t += 1
        }

      case FloatType =>
        output.asInstanceOf[Tensor[Float]].resizeAs(input.asInstanceOf[Tensor[Float]])
        val inputFloat = input.asInstanceOf[Tensor[Float]]
        val inputData = inputFloat.storage().array()
        val inputOffset = inputFloat.storageOffset()
        val outputFloat = output.asInstanceOf[Tensor[Float]]
        val outputData = outputFloat.storage().array()
        val outputOffset = outputFloat.storageOffset()

        var t = 0
        while (t < taskArray.length) {
          val _t = t
          results(_t) = Engine.model.invoke( () => {
            var i = taskArray(_t)._1
            while (i < taskArray(_t)._2) {
              outputData(outputOffset + i) = math.max(inputData(inputOffset + i), 0.0f)
              i += 1
            }
          })
          t += 1
        }
      case _ =>
        throw new UnsupportedOperationException("Only Float and Double type are supported")
    }

    Engine.model.sync(results)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val threadNum = Engine.model.getPoolSize
    val taskSize = gradOutput.nElement() / threadNum
    var extraTaskSize = gradOutput.nElement() % threadNum
    var allocated = 0
    val tasks = new ArrayBuffer[(Int, Int)]()
    while (allocated < gradOutput.nElement()) {
      val end = {
        if (extraTaskSize > 0) {
          extraTaskSize -= 1
          allocated + taskSize + 1
        } else {
          allocated + taskSize
        }
      }
      tasks += ((allocated, math.min(gradOutput.nElement(), end)))
      allocated = end
    }
    val taskArray = tasks.toArray
    val results = new Array[Future[Unit]](taskArray.length)

    ev.getType match {
      case DoubleType =>
        gradInput.asInstanceOf[Tensor[Double]].resizeAs(gradOutput.asInstanceOf[Tensor[Double]])
        gradInput.asInstanceOf[Tensor[Double]].copy(gradOutput.asInstanceOf[Tensor[Double]])
        val gradInputDouble = gradInput.asInstanceOf[Tensor[Double]]
        val inputDouble = input.asInstanceOf[Tensor[Double]]
        val gradInputData = gradInputDouble.storage().array()
        val gradInputOffset = gradInputDouble.storageOffset()
        val inputData = inputDouble.storage().array()
        val inputOffset = inputDouble.storageOffset()

        var t = 0
        while (t < taskArray.length) {
          val _t = t
          results(_t) = Engine.model.invoke(() => {
            var i = taskArray(_t)._1
            while (i < taskArray(_t)._2) {
              if (inputData(inputOffset + i) <= 0.0) {
                gradInputData(gradInputOffset + i) = 0.0
              }
              i += 1
            }
          })
          t += 1
        }
      case FloatType =>
        gradInput.asInstanceOf[Tensor[Float]].resizeAs(gradOutput.asInstanceOf[Tensor[Float]])
        gradInput.asInstanceOf[Tensor[Float]].copy(gradOutput.asInstanceOf[Tensor[Float]])
        val gradInputFloat = gradInput.asInstanceOf[Tensor[Float]]
        val inputFloat = input.asInstanceOf[Tensor[Float]]
        val gradInputData = gradInputFloat.storage().array()
        val gradInputOffset = gradInputFloat.storageOffset()
        val inputData = inputFloat.storage().array()
        val inputOffset = inputFloat.storageOffset()

        var t = 0
        while (t < taskArray.length) {
          val _t = t
          results(_t) = Engine.model.invoke(() => {
            var i = taskArray(_t)._1
            while (i < taskArray(_t)._2) {
              if (inputData(inputOffset + i) <= 0.0f) {
                gradInputData(gradInputOffset + i) = 0.0f
              }
              i += 1
            }
          })
          t += 1
        }
      case _ =>
        throw new UnsupportedOperationException("Only Float and Double type are supported")
    }

    Engine.model.sync(results)
    gradInput
  }

  override def clearState(): this.type = {
    super.clearState()
    this
  }
}

object ReLU {
  def apply[@specialized(Float, Double) T: ClassTag]()(implicit ev: TensorNumeric[T]): ReLU[T] = {
    new ReLU[T]()
  }
}
