package com.mass.scalann.tensor

import scala.reflect.ClassTag

trait Storage[T] extends Iterable[T] with Serializable {

  def length: Int

  override def size: Int = length

  def apply(index: Int): T

  def update(index: Int, value: T): Unit

  def copy(source: Storage[T], offset: Int, sourceOffset: Int, length: Int): this.type

  def copy(source: Storage[T]): this.type = copy(source, 0, 0, length)

  def fill(value: T, offset: Int, length: Int): this.type

  def resize(size: Long): this.type

  def array(): Array[T]

  def set(other: Storage[T]): this.type
}

object Storage {
  def apply[T: ClassTag](): Storage[T] = new ArrayStorage[T](new Array[T](0))

  def apply[T: ClassTag](size: Int): Storage[T] = new ArrayStorage[T](new Array[T](size))

  def apply[@specialized(Float, Double) T: ClassTag](
    data: Array[T]): Storage[T] = new ArrayStorage[T](data)
}
