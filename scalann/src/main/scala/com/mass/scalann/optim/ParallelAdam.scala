package com.mass.scalann.optim

import scala.reflect.ClassTag

import com.mass.scalann.tensor.{Tensor, TensorNumeric}
import com.mass.scalann.utils.{Engine, Table}

class ParallelAdam[@specialized(Float, Double) T: ClassTag](
    var learningRate: Double = 1e-3,
    var learningRateDecay: Double = 0.0,
    var beta1: Double = 0.9,
    var beta2: Double = 0.999,
    var epsilon: Double = 1e-8,
    var parallelNum: Int = Engine.coreNumber())(implicit ev: TensorNumeric[T])
    extends OptimMethod[T] {

  @transient private var ones: Tensor[T] = _

  override def optimize(
      feval: Tensor[T] => (T, Tensor[T]),
      parameter: Tensor[T]): (Tensor[T], Array[T]) = {
    val lr = this.learningRate
    val lrd = this.learningRateDecay
    val beta1 = this.beta1
    val beta2 = this.beta2
    val eps = this.epsilon

    val (fx, dfdx) = feval(parameter)
    var timestep = state.getOrElse[Int]("evalCounter", 0)
    val clr = lr / (1 + timestep * lrd)
    timestep += 1

    val gradLength = parameter.nElement()
    val taskSize = gradLength / parallelNum
    val extraTask = gradLength % parallelNum
    if (ones == null || ones.nElement() < taskSize + 1) {
      ones = Tensor[T]().resize(taskSize + 1).fill(ev.one)
    }

    (0 until parallelNum).foreach{tid =>
      if (state.get[Tensor[T]](s"s$tid").isEmpty) {
        state(s"s$tid") = Tensor[T]()
        state(s"r$tid") = Tensor[T]()
        state(s"denom$tid") = Tensor[T]()
      }
    }

    Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
      val offset = tid * taskSize + math.min(tid, extraTask)
      val length = taskSize + (if (tid < extraTask) 1 else 0)
      val curDfdx = dfdx.narrow(0, offset, length)
      val curParam = parameter.narrow(0, offset, length)
      val curOnes = ones.narrow(0, 0, length)
      val _s = state.get[Tensor[T]](s"s$tid").get.resizeAs(curParam)
      val _r = state.get[Tensor[T]](s"r$tid").get.resizeAs(curParam)
      val _denom = state.get[Tensor[T]](s"denom$tid").get.resizeAs(curParam)
      ParallelAdam.updateFrame(_s, _r, _denom, clr, curDfdx, curParam, beta1,
        beta2, timestep, curOnes, eps)
    }))

    state("evalCounter") = timestep
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

object ParallelAdam {
  private[optim] def updateFrame[T: ClassTag](_s: Tensor[T], _r: Tensor[T], _denom: Tensor[T],
      clr: Double, dfdx: Tensor[T], parameter: Tensor[T], beta1: Double, beta2: Double,
      timestep: Int, ones: Tensor[T], eps: Double)(implicit ev: TensorNumeric[T]): Unit = {

    _s.mul(ev.fromType[Double](beta1)).add(ev.fromType[Double](1 - beta1), dfdx)
    _denom.cmul(dfdx, dfdx)
    _r.mul(ev.fromType[Double](beta2)).add(ev.fromType[Double](1 - beta2), _denom)
    _denom.sqrt(_r)
    _denom.add(ev.fromType(eps), ones)

    val biasCorrection1 = 1 - math.pow(beta1, timestep)
    val biasCorrection2 = 1 - math.pow(beta2, timestep)
    val stepSize = clr * math.sqrt(biasCorrection2) / biasCorrection1
    _denom.cdiv(_s, _denom)
    parameter.add(ev.fromType[Double](-stepSize), _denom)
  }
}
