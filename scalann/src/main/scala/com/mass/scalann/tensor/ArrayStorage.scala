package com.mass.scalann.tensor

import java.util

import scala.reflect._

private[tensor] class ArrayStorage[@specialized(Double, Float) T: ClassTag](
    private[tensor] var values: Array[T]
) extends Storage[T] {

  override def apply(index: Int): T = values(index)

  override def update(index: Int, value: T): Unit = values(index) = value

  override def length: Int = values.length

  override def iterator: Iterator[T] = values.iterator

  override def array(): Array[T] = values

  override def copy(
      source: Storage[T],
      offset: Int,
      sourceOffset: Int,
      length: Int
  ): ArrayStorage.this.type = {
    source match {
      case s: ArrayStorage[T] =>
        System.arraycopy(s.values, sourceOffset, this.values, offset, length)
      case _ => throw new UnsupportedOperationException("Only support dnn or array storage")
    }
    this
  }

  override def resize(size: Long): this.type = {
    values = new Array[T](size.toInt)
    this
  }

  override def fill(value: T, offset: Int, length: Int): this.type = {

    value match {
      case v: Double =>
        util.Arrays.fill(values.asInstanceOf[Array[Double]], offset, offset + length, v)
      case v: Float =>
        util.Arrays.fill(values.asInstanceOf[Array[Float]], offset, offset + length, v)
      case v: Int => util.Arrays.fill(values.asInstanceOf[Array[Int]], offset, offset + length, v)
      case v: Long => util.Arrays.fill(values.asInstanceOf[Array[Long]], offset, offset + length, v)
      case v: Short =>
        util.Arrays.fill(values.asInstanceOf[Array[Short]], offset, offset + length, v)
      case _ => throw new IllegalArgumentException
    }

    this
  }

  override def set(other: Storage[T]): this.type = {
    require(other.length == this.length)
    this.values = other.array()
    this
  }
}
