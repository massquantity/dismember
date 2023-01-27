package com.mass.scalann.nn

import scala.reflect.ClassTag

import com.mass.scalann.nn.abstractnn.AbstractCriterion
import com.mass.scalann.tensor.{Tensor, TensorNumeric}

class ClassNLLCriterion[@specialized(Float, Double) T: ClassTag](
    sizeAverage: Boolean = true,
    logProbAsInput: Boolean = true
)(implicit ev: TensorNumeric[T])
    extends AbstractCriterion[T] {

  private val lowerBound: T = ev.fromType(1e-8)
  private val upperBound: T = ev.minus(ev.one, lowerBound)

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    val nClasses = input.size(input.dim() - 1)
    if (input.dim() == 1) {
      val newTarget = {
        if (target.dim() == 2 && target.size(0) == 1) {
          target.clone().squeeze()
        } else {
          target
        }
      }
      require(input.dim() == newTarget.dim())
      val curTarget = ev.toType[Int](newTarget.valueAt(0))
      assert(curTarget >= 0 && curTarget < nClasses, s"label should be in range [0, $nClasses)")
      output = {
        if (!logProbAsInput) {
          val clipped = ev.clip(input.valueAt(curTarget), lowerBound, upperBound)
          ev.negative(ev.log(clipped))
        } else {
          ev.negative(input.valueAt(curTarget))
        }
      }
    } else if (input.dim() == 2) {
      val batchSize = input.size(0)
      val targetSize = target.size()
      target.squeeze()
      require(target.dim() == 1, "label should be 1D tensor")
      output = ev.fromType[Int](0)
      var i = 0
      while (i < batchSize) {
        val curTarget = ev.toType[Int](target.valueAt(i))
        assert(curTarget >= 0 && curTarget < nClasses, s"label should be in range [0, $nClasses)")
        val loss =
          if (!logProbAsInput) {
            val clipped = ev.clip(input.valueAt(i, curTarget), lowerBound, upperBound)
            ev.log(clipped)
          } else {
            input.valueAt(i, curTarget)
          }
        output = ev.minus(output, loss)
        i += 1
      }
      target.resize(targetSize)
    }

    if (sizeAverage && input.dim() == 2) {
      output = ev.divide(output, ev.fromType(input.size(0)))
    }
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).zero()
    if (input.dim() == 1) {
      require(input.dim() == target.dim())
      val curTarget = ev.toType[Int](target.valueAt(0))
      gradInput.setValue(curTarget, ev.fromType[Int](-1))
      if (!logProbAsInput) {
        val clipped = ev.clip(input.valueAt(curTarget), lowerBound, upperBound)
        gradInput.setValue(curTarget, ev.times(gradInput.valueAt(curTarget), ev.inv(clipped)))
      }
    } else if (input.dim() == 2) {
      val batchSize = input.size(0)
      val targetSize = target.size()
      target.squeeze()
      var i = 0
      while (i < batchSize) {
        val curTarget = ev.toType[Int](target.valueAt(i))
        gradInput.setValue(i, curTarget, ev.fromType[Int](-1))
        if (sizeAverage) {
          val newGrad = ev.divide(gradInput.valueAt(i, curTarget), ev.fromType(batchSize))
          gradInput.setValue(i, curTarget, newGrad)
        }
        if (!logProbAsInput) {
          val clipped = ev.clip(input.valueAt(i, curTarget), lowerBound, upperBound)
          val newGrad = ev.times(gradInput.valueAt(i, curTarget), ev.inv(clipped))
          gradInput.setValue(i, curTarget, newGrad)
        }
        i += 1
      }
      target.resize(targetSize)
    }
    gradInput
  }
}

object ClassNLLCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
      sizeAverage: Boolean = true,
      logProbAsInput: Boolean = true
  )(implicit ev: TensorNumeric[T]): ClassNLLCriterion[T] = {
    new ClassNLLCriterion[T](sizeAverage, logProbAsInput)
  }
}
