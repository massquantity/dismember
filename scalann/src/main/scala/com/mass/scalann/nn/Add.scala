package com.mass.scalann.nn

import scala.reflect.ClassTag

import com.mass.scalann.nn.abstractnn.AbstractModule
import com.mass.scalann.tensor.{Tensor, TensorNumeric}
import com.mass.scalann.utils.Table

// element-wise add
class Add[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Tensor[T], T] {

  override def updateOutput(input: Table): Tensor[T] = {
    output.resizeAs(input(0)).copy(input(0))
    var i = 1
    while (i < input.length) {
      require(input[Tensor[T]](i).isSameSizeAs(input[Tensor[T]](0)), "all inputs must have same size")
      output.add(input[Tensor[T]](i))
      i += 1
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    var i = 0
    while (i < input.length) {
      if (i >= gradInput.length) {
        gradInput.insert(i, Tensor[T]())
      }
      require(input[Tensor[T]](i).isSameSizeAs(gradOutput), "input must have same size as output")
      gradInput[Tensor[T]](i).resizeAs(gradOutput).copy(gradOutput)
      i += 1
    }
    gradInput
  }

  override def clearState(): this.type = {
    super.clearState()
    this
  }
}

object Add {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): Add[T] = {
    new Add[T]()
  }
}
