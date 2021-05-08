package com.mass.sparkdl.parameters

import java.nio.ByteBuffer

import scala.reflect.ClassTag

import com.mass.sparkdl.tensor.Tensor

trait CompressedTensor[T] extends Serializable {

  def bytes(offset: Int, length: Int): ByteBuffer

  def bytes(): ByteBuffer

  def compress(offset: Int, src: Tensor[T], srcOffset: Int, length: Int): this.type

  def compress(tensor: Tensor[T]): this.type

  def deCompress(srcOffset: Int, tensor: Tensor[T], tgtOffset: Int, length: Int): Unit

  def deCompress(tensor: Tensor[T]): Unit

  def add(data: ByteBuffer, offset: Int, length: Int): this.type

  def add(data: ByteBuffer): this.type

  def parAdd(data: ByteBuffer, offset: Int, length: Int): this.type

  def parAdd(data: ByteBuffer): this.type
}

object SerializerInstance {
  def serialize[T: ClassTag](data: Tensor[T], pm: String = "fp16"): CompressedTensor[T] = {
    new FP16CompressedTensor[T](data)
  }

  def create[T: ClassTag](length: Int, pm: String): CompressedTensor[T] = {
    new FP16CompressedTensor[T](length)
  }

  def create[T: ClassTag](data: ByteBuffer, pm: String = "fp16"): CompressedTensor[T] = {
    new FP16CompressedTensor[T](data)
  }
}
