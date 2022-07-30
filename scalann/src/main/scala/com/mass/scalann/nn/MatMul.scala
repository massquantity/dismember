package com.mass.scalann.nn

import scala.reflect.ClassTag

import com.mass.scalann.nn.abstractnn.AbstractModule
import com.mass.scalann.tensor.{Tensor, TensorNumeric}
import com.mass.scalann.utils.Table

class MatMul[T: ClassTag](transB: Boolean = false)(implicit ev: TensorNumeric[T])
    extends AbstractModule[Table, Tensor[T], T] {

  gradInput = Table(Tensor[T](), Tensor[T]())

  override def updateOutput(input: Table): Tensor[T] = {
    var (ma, mb) = (input[Tensor[T]](0), input[Tensor[T]](1))

    if (ma.dim() == 2) {
      require(mb.dim() == 2, s"second input tensor must be 2D, got ${mb.dim()}")

      if (transB) {
        mb = mb.t()
      }
      require(ma.size(1) == mb.size(0), "matrix size doesn't match")

      output.resize(ma.size(0), mb.size(1))
      output.addmm(ev.zero, ev.one, ma, mb)
    } else {
      require(ma.dim() == mb.dim(), s"input tensors should be with same dimension," +
        s"but got ${ma.dim()} ${mb.dim()}")

      if (transB) {
        mb = mb.transpose(1, 2)
      }
      require(ma.size(2) == mb.size(1), "matrix size doesn't match")

      output.resize(ma.size(0), ma.size(1), mb.size(2)).zero()
      output.bmm(ev.zero, ev.one, ma, mb)
    }

    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    require(gradOutput.isContiguous, "gradOutput must be contiguous")
    val (ma, mb) = (input[Tensor[T]](0), input[Tensor[T]](1))
    gradInput[Tensor[T]](0).resizeAs(ma).zero()
    gradInput[Tensor[T]](1).resizeAs(mb).zero()

    if (ma.dim() == 2) {
      if (!transB) {
        gradInput[Tensor[T]](0).addmm(ev.zero, ev.one, gradOutput, mb.t())
        gradInput[Tensor[T]](1).addmm(ev.zero, ev.one, ma.t(), gradOutput)
      } else {
        gradInput[Tensor[T]](0).addmm(ev.zero, ev.one, gradOutput, mb)
        gradInput[Tensor[T]](1).addmm(ev.zero, ev.one, gradOutput.t(), ma)
      }
    } else {
      if (!transB) {
        gradInput[Tensor[T]](0).bmm(ev.zero, ev.one, gradOutput, mb.transpose(1, 2))
        gradInput[Tensor[T]](1).bmm(ev.zero, ev.one, ma.transpose(1, 2), gradOutput)
      } else {
        gradInput[Tensor[T]](0).bmm(ev.zero, ev.one, gradOutput, mb)
        gradInput[Tensor[T]](1).bmm(ev.zero, ev.one, gradOutput.transpose(1, 2), ma)
      }
    }
    gradInput
  }

  override def clearState(): this.type = {
    super.clearState()
    gradInput[Tensor[T]](0).set()
    gradInput[Tensor[T]](1).set()
    this
  }
}

object MatMul {
  def apply[@specialized(Float, Double) T: ClassTag](transB: Boolean = false)(
      implicit ev: TensorNumeric[T]): MatMul[T] = {
    new MatMul[T](transB)
  }
}
