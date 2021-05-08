package com.mass.sparkdl.nn

import scala.concurrent.{Await, Future}
import scala.concurrent.duration.Duration
import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.AbstractCriterion
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}
import com.mass.sparkdl.utils.Engine

class ClassNLLCriterion[@specialized(Float, Double) T: ClassTag](sizeAverage: Boolean = true,
    logProbAsInput: Boolean = true)(implicit ev: TensorNumeric[T]) extends AbstractCriterion[T] {

  @transient var results: Array[Future[T]] = _
  @transient var resultsBackward: Array[Future[_]] = _
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
      if (results == null || results.length != batchSize) {
        results = new Array[Future[T]](batchSize)
      }

      var i = 0
      while (i < batchSize) {
        val _i = i
        results(_i) = Engine.model.invoke( () => {
          val curTarget = ev.toType[Int](target.valueAt(_i))
          assert(curTarget >= 0 && curTarget < nClasses, s"label should be in range [0, $nClasses)")
          if (!logProbAsInput) {
            val clipped = ev.clip(input.valueAt(_i, curTarget), lowerBound, upperBound)
            ev.log(clipped)
          } else {
            input.valueAt(_i, curTarget)
          }
        })
        i += 1
      }

      output = ev.fromType[Int](0)
      i = 0
      while (i < batchSize) {
        val loss = Await.result(results(i), Duration.Inf)
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
      if (resultsBackward == null || resultsBackward.length != batchSize) {
        resultsBackward = new Array[Future[_]](batchSize)
      }

      var i = 0
      while (i < batchSize) {
        val _i = i
        resultsBackward(_i) = Engine.model.invoke( () => {
          val curTarget = ev.toType[Int](target.valueAt(_i))
          gradInput.setValue(_i, curTarget, ev.fromType[Int](-1))
          if (sizeAverage) {
            val newGrad = ev.divide(gradInput.valueAt(_i, curTarget), ev.fromType(batchSize))
            gradInput.setValue(_i, curTarget, newGrad)
          }
          if (!logProbAsInput) {
            val clipped = ev.clip(input.valueAt(_i, curTarget), lowerBound, upperBound)
            val newGrad = ev.times(gradInput.valueAt(_i, curTarget), ev.inv(clipped))
            gradInput.setValue(_i, curTarget, newGrad)
          }
        })
        i += 1
      }

      i = 0
      while (i < batchSize) {
        Await.result(resultsBackward(i), Duration.Inf)
        i += 1
      }
      target.resize(targetSize)
    }
    gradInput
  }
}

object ClassNLLCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](sizeAverage: Boolean = true,
      logProbAsInput: Boolean = true)(implicit ev: TensorNumeric[T]): ClassNLLCriterion[T] = {
    new ClassNLLCriterion[T](sizeAverage, logProbAsInput)
  }
}
