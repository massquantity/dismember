package com.mass.scalann.optim

import scala.reflect.ClassTag

import com.mass.scalann.tensor.{Tensor, TensorNumeric}
import com.mass.scalann.utils.Table

class Adam[@specialized(Float, Double) T: ClassTag](
    var learningRate: Double = 1e-3,
    var learningRateDecay: Double = 0.0,
    var beta1: Double = 0.9,
    var beta2: Double = 0.999,
    var epsilon: Double = 1e-8)(implicit ev: TensorNumeric[T]) extends OptimMethod[T] {

  @transient private var buffer: Tensor[T] = _

  override def optimize(
      feval: Tensor[T] => (T, Tensor[T]),
      parameter: Tensor[T]): (Tensor[T], Array[T]) = {

    if (buffer == null) {
      buffer = Tensor[T]()
    }
    val lr = this.learningRate
    val lrd = this.learningRateDecay
    val beta1 = this.beta1
    val beta2 = this.beta2
    val eps = this.epsilon

    val (fx, dfdx) = feval(parameter)
    var timestep = state.getOrElse[Int]("trainCounter", 0)
    val (_s, _r, _denom) = {
      if (state.get[Tensor[T]]("s").isDefined) {
        (state.get[Tensor[T]]("s").get,
         state.get[Tensor[T]]("r").get,
         state.get[Tensor[T]]("denom").get.resizeAs(dfdx))
      } else {
        (Tensor[T]().resizeAs(dfdx).zero(),
         Tensor[T]().resizeAs(dfdx).zero(),
         Tensor[T]().resizeAs(dfdx).zero())
      }
    }

    val clr = lr / (1 + timestep * lrd)
    timestep += 1

    _s.mul(ev.fromType[Double](beta1)).add(ev.fromType[Double](1 - beta1), dfdx)
    buffer.resizeAs(dfdx).cmul(dfdx, dfdx)
    _r.mul(ev.fromType[Double](beta2)).add(ev.fromType[Double](1 - beta2), buffer)
    _denom.sqrt(_r)
    buffer.fill(ev.one)
    _denom.add(ev.fromType[Double](eps), buffer)

    val biasCorrection1 = 1 - math.pow(beta1, timestep)
    val biasCorrection2 = 1 - math.pow(beta2, timestep)
    val stepSize = clr * math.sqrt(biasCorrection2) / biasCorrection1
    _denom.cdiv(_s, _denom)
    parameter.add(ev.fromType[Double](-stepSize), _denom)
    // parameter.addcdiv(ev.fromType[Double](-stepSize), _s, _denom)

    state("trainCounter") = timestep
    state("s") = _s
    state("r") = _r
    state("denom") = _denom
    (parameter, Array(fx))
  }

  override def loadFromTable(config: Table): this.type = {
    this.learningRate = config.get[Double]("learningRate").getOrElse(this.learningRate)
    this.learningRateDecay = config.get[Double]("learningRateDecay").getOrElse(this.learningRateDecay)
    this.beta1 = config.get[Double]("beta1").getOrElse(this.beta1)
    this.beta2 = config.get[Double]("beta2").getOrElse(this.beta2)
    this.epsilon = config.get[Double]("Epsilon").getOrElse(this.epsilon)
    this
  }

  override def clearHistory(): Unit = {
    state.delete("s")
    state.delete("r")
  }

  override def getLearningRate: Double = this.learningRate
}

object Adam {
  def apply[@specialized(Float, Double) T: ClassTag](
      learningRate: Double,
      learningRateDecay: Double = 0.0,
      beta1: Double = 0.9,
      beta2: Double = 0.999,
      epsilon: Double = 1e-8)(implicit ev: TensorNumeric[T]): Adam[T] = {
    new Adam[T](learningRate, learningRateDecay, beta1, beta2, epsilon)
  }
}
