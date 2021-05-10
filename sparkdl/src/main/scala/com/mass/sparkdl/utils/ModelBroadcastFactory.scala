package com.mass.sparkdl.utils

import scala.reflect.ClassTag

import com.mass.sparkdl.tensor.TensorNumeric

trait ModelBroadcastFactory {
  def create[T: ClassTag]()(implicit ev: TensorNumeric[T]) : ModelBroadcast[T]
}

private[sparkdl] class DefaultModelBroadcastFactory extends ModelBroadcastFactory {
  override def create[T: ClassTag]()(implicit ev: TensorNumeric[T]): ModelBroadcast[T] = {
    new ModelBroadcastImp[T]()
  }
}

private[sparkdl] class ProtoBufferModelBroadcastFactory extends ModelBroadcastFactory {
  override def create[T: ClassTag]()(implicit ev: TensorNumeric[T]): ModelBroadcast[T] = {
    new ModelBroadcastImp[T](true)
  }
}
