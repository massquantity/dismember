package com.mass.sparkdl.nn.mixin

import scala.reflect.ClassTag

import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}

object Module {

  def flatten[@specialized(Float, Double) T: ClassTag](parameters: Array[Tensor[T]])(
      implicit ev: TensorNumeric[T]): Tensor[T] = {
    val compactedTensor = isCompact(parameters)
    if (compactedTensor != null) {
      return compactedTensor
    }

    var i = 0
    var length = 0
    while (i < parameters.length) {
      require(parameters(i).isContiguous, "parameters should be contiguous")
      length += parameters(i).nElement()
      i += 1
    }

    val result = Tensor[T](length)
    val resultStorage = result.storage()

    i = 0
    var offset = 0
    // by setting parameters close together in resultStorage, the parameters in model become compact.
    while (i < parameters.length) {
      System.arraycopy(parameters(i).storage().array(), parameters(i).storageOffset(),
        resultStorage.array(), offset, parameters(i).nElement())
      parameters(i).set(resultStorage, offset, parameters(i).size(), parameters(i).stride())
      offset += parameters(i).nElement()
      i += 1
    }

    result
  }

  def isCompact[@specialized(Float, Double) T: ClassTag](parameters: Array[Tensor[T]])(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(parameters.length > 0,
      "The length of paramters should >= 0" +
        "parameter length" +
        s" ${parameters.length}")
    var i = 1
    val storage = parameters(0).storage()
    var length = parameters(0).nElement()
    val offset = parameters(0).storageOffset()
    // make sure parameters is shared and contiguous
    while (i < parameters.length) {
      if (!storage.eq(parameters(i).storage())) {
        return null
      }
      if (offset + length != parameters(i).storageOffset()) {
        return null
      }
      length += parameters(i).nElement()
      i += 1
    }

    Tensor(storage, offset, Array(length))
  }
}
