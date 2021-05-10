package com.mass.sparkdl.utils

import scala.collection.mutable
import scala.collection.Set

import com.mass.sparkdl.nn.abstractnn.Activity
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}

class Table(
    private val state: mutable.Map[Any, Any] = new mutable.HashMap[Any, Any](),
    private var topIndex: Int = 0) extends Serializable with Activity {

  private[sparkdl] def this(data: Array[Any]) = {
    this(new mutable.HashMap[Any, Any](), 0)
    while (topIndex < data.length) {
      state.put(topIndex, data(topIndex))
      topIndex += 1
    }
  }

  override def isTensor: Boolean = false

  override def isTable: Boolean = true

  private[sparkdl] def getState: Map[Any, Any] = {
    state.toMap
  }

  def clear(): this.type = {
    state.clear()
    topIndex = 0
    this
  }

  def length: Int = state.size

  def keySet: Set[Any] = state.keySet

  def foreach[U](func: ((Any, Any)) => U): Unit = state.foreach(func)

  def map[U](func: ((Any, Any)) => U): Iterable[U] = state.map(func)

  def get[T](key: Any): Option[T] = {
    state.get(key).map(_.asInstanceOf[T])
  }

  def getOrElse[T](key: Any, default: T): T = {
    state.getOrElse(key, default).asInstanceOf[T]
  }

  def contains(key: Any): Boolean = state.contains(key)

  def apply[T](key: Any): T = {
    state(key).asInstanceOf[T]
  }

  def update(key: Any, value: Any): this.type = {
    state(key) = value
    key match {
      case i: Int if topIndex == i =>
        topIndex += 1
        while (state.contains(topIndex)) {
          topIndex += 1
        }
      case _ =>
    }
    this
  }

  override def clone(): Table = {
    val result = new Table()
    for (k <- state.keys) {
      result(k) = state(k)
    }
    result
  }

  def delete(obj: Any): this.type = {
    if (state.get(obj).isDefined) {
      state.remove(obj)
    }
    this
  }

  def insert[T](obj: T): this.type = update(topIndex, obj)

  def insert[T](index: Int, obj: T): this.type = {
    require(index >= 0)

    if (topIndex > index) {
      var i = topIndex
      topIndex += 1
      while (i > index) {
        state(i) = state(i - 1)
        i -= 1
      }
      update(index, obj)
    } else {
      update(index, obj)
    }

    this
  }

  override def toTensor[D](implicit ev: TensorNumeric[D]): Tensor[D] =
    throw new IllegalArgumentException("Table cannot be cast to Tensor")

  override def toTable: Table = this
}

object T {
  def apply(): Table = new Table()

  def apply(data1: Any, datas: Any*): Table = {
    val firstElement = Array(data1)
    val otherElements = datas.toArray
    new Table(firstElement ++ otherElements)
  }

  def array(data: Array[_]): Table = {
    new Table(data.asInstanceOf[Array[Any]])
  }

  def seq(data: Seq[_]): Table = {
    new Table(data.toArray.asInstanceOf[Array[Any]])
  }

  def apply(tuple: (Any, Any), tuples: (Any, Any)*): Table = {
    val table = new Table()
    table(tuple._1) = tuple._2
    for ((k, v) <- tuples) {
      table(k) = v
    }
    table
  }

  def load(path : String) : Table = {
    File.load[Table](path)
  }
}
