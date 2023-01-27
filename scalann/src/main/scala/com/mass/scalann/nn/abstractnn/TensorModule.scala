package com.mass.scalann.nn.abstractnn

import scala.reflect.ClassTag

import com.mass.scalann.tensor.{Tensor, TensorNumeric}

abstract class TensorModule[T: ClassTag](implicit
    ev: TensorNumeric[T]
) extends AbstractModule[Tensor[T], Tensor[T], T]
