package com.mass.sparkdl.nn

import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.TensorModule
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}

// -1 means inferring shape of one dimension,
// also assumes first dimension is batch size, thus excluded.
class Reshape[T: ClassTag](val size: Array[Int])(
    implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  require(size.count(_ != -1) >= size.length - 1, "only one dim is allowed to infer")
  private var inplace: Boolean = true
  private var newSize: Array[Int] = _

  private def checkSize(inputSize: Array[Int]): Unit = {
    // add batch size
    newSize = Array(0) ++ size.clone()
    if (size.contains(-1)) {
      val totalSize = inputSize.product
      val concreteSize = size.filter(_ != -1).product
      require(totalSize % concreteSize == 0, "size doesn't match")
      val index = size.indexOf(-1) + 1
      newSize(index) = totalSize / concreteSize
    } else {
      require(size.product == inputSize.product,
        s"reshape size must match original total size, " +
          s"reshape: ${size.mkString(" ")}, " +
          s"inputSize: ${inputSize.mkString(" ")}")
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (newSize == null) {
      checkSize(input.size().slice(1, input.dim()))
    }
    newSize(0) = input.size(0)

    if (input.isContiguous) {
      output = input.view(newSize)
    } else {
      output = input.contiguous().view(newSize)
      inplace = false
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (gradOutput.isContiguous) {
      gradInput = gradOutput.view(input.size())
    } else {
      gradInput = gradOutput.contiguous().view(input.size())
    }
    gradInput
  }

  override def clearState(): this.type = {
    if (!inplace) {
      super.clearState()
    }
    this
  }
}

object Reshape {
  def apply[T: ClassTag](size: Array[Int])(implicit ev: TensorNumeric[T]): Reshape[T] = {
    new Reshape[T](size)
  }
}
