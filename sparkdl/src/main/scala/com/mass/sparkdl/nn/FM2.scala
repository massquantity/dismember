package com.mass.sparkdl.nn

import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.TensorModule
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}

class FM2[T: ClassTag](merge: Boolean = true)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  private var buffer = Tensor[T]()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 3, "FM input must be 3D")
    val batchSize = input.size(0)
    val embedSize = input.size(2)
    buffer = input.sum(1)  // m * 1 * k
    val sumSquare = Tensor[T]().resizeAs(buffer).copy(buffer).pow(ev.fromType(2.0))
    val squareSum = Tensor[T]().resizeAs(input).copy(input).pow(ev.fromType(2.0)).sum(1)
    output.resize(Array(batchSize, 1, embedSize))
    if (merge) {
      output = output.add(sumSquare).sub(squareSum).sum(2).squeeze().div(ev.fromType(2.0))
      require(output.dim() == 1, "merged FM output must be 1D")
    } else {
      output = output.add(sumSquare).sub(squareSum).squeeze().div(ev.fromType(2.0))
      require(output.dim() == 2, "unmerged FM output must be 2D")
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val batchSize = input.size(0)
    val featSize = input.size(1)
    val embedSize = input.size(2)
    gradInput = gradOutput
      .reshape(Array(batchSize, 1, 1))
      .repeatTensor(Array(1, featSize, embedSize))
    val gradSumSquare = buffer.repeatTensor(Array(1, featSize, 1))
    gradInput.cmul(gradSumSquare.sub(input))
    gradInput
  }

  def updateGradInput2(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).zero()
    val batchSize = input.size(0)
    val featSize = input.size(1)
    val embedSize = input.size(2)
    val gradInputData = gradInput.storage().array()
    val bufferData = buffer.storage().array()
    val inputData = input.storage().array()
    val gradOutputData = gradOutput.storage().array()
    var m_stride = 0
    var n_stride = 0
    var k_stride = 0
    var m = 0
    while (m < batchSize) {
      var n = 0
      while (n < featSize) {
        var k = 0
        while (k < embedSize) {
          gradInputData(m_stride + k) =
            ev.plus(gradInputData(m_stride + k),
              ev.times(gradOutputData(m),
                ev.minus(bufferData(n_stride + k), inputData(m_stride + k))
              )
            )
          k += 1
        }
        m_stride += embedSize
        n += 1
      }
      n_stride += embedSize
      m += 1
    }
    gradInput
  }
}
