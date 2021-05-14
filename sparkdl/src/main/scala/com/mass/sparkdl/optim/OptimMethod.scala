package com.mass.sparkdl.optim

import scala.reflect.ClassTag

import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.utils.{File, T, Table}
import org.apache.commons.lang3.SerializationUtils

trait OptimMethod[@specialized(Float, Double) T] extends Serializable {

  var state: Table = T(
    "epoch" -> 0,
    "trainIter" -> 0)

  def optimize(feval: Tensor[T] => (T, Tensor[T]), parameter: Tensor[T]): (Tensor[T], Array[T])

  def clearHistory(): Unit

  def updateHyperParameter(): Unit = { }

  def getHyperParameter: String = ""

  override def clone(): OptimMethod[T] = SerializationUtils.clone(this)

  def getLearningRate: Double

  def updateState(key: Any, value: Any): Unit = {
    state.update(key, value)
  }

  def save(path: String, overWrite: Boolean = false): this.type = {
    this.clearHistory()
    File.save(this, path, overWrite)
    this
  }

  def loadFromTable(config: Table): this.type
}

object OptimMethod {

  def load[T: ClassTag](path : String) : OptimMethod[T] = {
    File.load[OptimMethod[T]](path)
  }
}
