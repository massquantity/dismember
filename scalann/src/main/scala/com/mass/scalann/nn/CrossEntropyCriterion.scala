package com.mass.scalann.nn

import scala.reflect.ClassTag

import com.mass.scalann.nn.abstractnn.AbstractCriterion
import com.mass.scalann.tensor.{Tensor, TensorNumeric}

class CrossEntropyCriterion[T: ClassTag](val sizeAverage: Boolean = true)(implicit
    ev: TensorNumeric[T]
) extends AbstractCriterion[T] {

  private val logSoftMax = new LogSoftMax[T]()
  private val crossEntropy = new ClassNLLCriterion[T](sizeAverage, logProbAsInput = true)

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    logSoftMax.updateOutput(input)
    crossEntropy.updateOutput(logSoftMax.output, target)
    output = crossEntropy.output
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    val _gradInput = crossEntropy.updateGradInput(logSoftMax.output, target)
    logSoftMax.updateGradInput(input, _gradInput)
    gradInput = gradInput.resizeAs(logSoftMax.gradInput).copy(logSoftMax.gradInput)
    gradInput
  }
}

object CrossEntropyCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
      sizeAverage: Boolean = true
  )(implicit ev: TensorNumeric[T]): CrossEntropyCriterion[T] = {
    new CrossEntropyCriterion[T](sizeAverage)
  }
}
