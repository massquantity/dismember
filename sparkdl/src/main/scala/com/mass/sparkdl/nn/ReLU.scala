package com.mass.sparkdl.nn

import scala.concurrent.Future
import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.TensorModule
import com.mass.sparkdl.tensor.{DoubleType, FloatType, Tensor, TensorNumeric}
import com.mass.sparkdl.utils.Engine

class ReLU[T: ClassTag](implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.isContiguous, "input must be contiguous")
    val results = new Array[Future[Unit]](1)

    ev.getType match {
      case DoubleType =>
        output.asInstanceOf[Tensor[Double]].resizeAs(input.asInstanceOf[Tensor[Double]])
        val inputDouble = input.asInstanceOf[Tensor[Double]]
        val inputData = inputDouble.storage().array()
        val inputOffset = inputDouble.storageOffset()
        val outputDouble = output.asInstanceOf[Tensor[Double]]
        val outputData = outputDouble.storage().array()
        val outputOffset = outputDouble.storageOffset()

        val end = input.nElement()
        results(0) = Engine.model.invoke( () => {
          var i = 0
          val _end = end
          while (i < _end) {
            outputData(outputOffset + i) = math.max(inputData(inputOffset + i), 0.0)
            i += 1
          }
        })

      case FloatType =>
        output.asInstanceOf[Tensor[Float]].resizeAs(input.asInstanceOf[Tensor[Float]])
        val inputFloat = input.asInstanceOf[Tensor[Float]]
        val inputData = inputFloat.storage().array()
        val inputOffset = inputFloat.storageOffset()
        val outputFloat = output.asInstanceOf[Tensor[Float]]
        val outputData = outputFloat.storage().array()
        val outputOffset = outputFloat.storageOffset()

        val end = input.nElement()
        results(0) = Engine.model.invoke( () => {
          var i = 0
          val _end = end
          while (i < _end) {
            outputData(outputOffset + i) = math.max(inputData(inputOffset + i), 0.0f)
            i += 1
          }
        })

      case _ =>
        throw new UnsupportedOperationException("Only Float and Double type are supported")
    }

    Engine.model.sync(results)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
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

        val results = Array[Future[Unit]] {
          Engine.model.invoke( () => {
            var i = 0
            val end = input.nElement()
            while (i < end) {
              if (inputData(inputOffset + i) <= 0.0) {
                gradInputData(gradInputOffset + i) = 0.0
              }
              i += 1
            }
          })
        }
        Engine.model.sync(results)

      case FloatType =>
        gradInput.asInstanceOf[Tensor[Float]].resizeAs(gradOutput.asInstanceOf[Tensor[Float]])
        gradInput.asInstanceOf[Tensor[Float]].copy(gradOutput.asInstanceOf[Tensor[Float]])
        val gradInputFloat = gradInput.asInstanceOf[Tensor[Float]]
        val inputFloat = input.asInstanceOf[Tensor[Float]]
        val gradInputData = gradInputFloat.storage().array()
        val gradInputOffset = gradInputFloat.storageOffset()
        val inputData = inputFloat.storage().array()
        val inputOffset = inputFloat.storageOffset()

        val results = Array[Future[Unit]] {
          Engine.model.invoke( () => {
            var i = 0
            val end = input.nElement()
            while (i < end) {
              if (inputData(inputOffset + i) <= 0.0f) {
                gradInputData(gradInputOffset + i) = 0.0f
              }
              i += 1
            }
          })
        }
        Engine.model.sync(results)

      case _ =>
        throw new UnsupportedOperationException("Only Float and Double type are supported")
    }

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
