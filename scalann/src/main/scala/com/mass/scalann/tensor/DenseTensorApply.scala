package com.mass.scalann.tensor

object DenseTensorApply {
  def apply1[@specialized(Float, Double) T](tensor: Tensor[T], func: TensorFunc2[T]): Unit = {
    if (tensor.isEmpty) {
      return
    }

    // shortcut for scalar
    if (tensor.isScalar) {
      val data = tensor.storage().array()
      val index = tensor.storageOffset()
      func(data, index)
      return
    }

    val stride = getStride(tensor)
    val (largestDim, largestSize) = getLargestContiguousSize(tensor)
    val counter = getCounter(largestDim)
    val data = tensor.storage().array()
    var offset = tensor.storageOffset()
    var hasFinished = false
    var i = 0
    while (!hasFinished) {
      while (i < largestSize) {
        val index = offset + i * stride
        func(data, index)
        i += 1
      }
      val r = updateCounter(tensor, counter, offset, largestDim)
      hasFinished = r._1
      offset = r._2
      i = 0
    }
  }

  def apply2[T](tensor1: Tensor[T], tensor2: Tensor[T],
    func: TensorFunc4[T]): Unit = {
    require(tensor1.nElement() == tensor2.nElement(),
      s"inconsistent tensor size: ${tensor1.nElement()} == ${tensor2.nElement()}")

    if (tensor1.isEmpty) {
      return
    }

    // shortcut for scalar
    if (tensor1.isScalar && tensor2.isScalar) {
      val tensor1Data = tensor1.storage().array()
      val tensor2Data = tensor2.storage().array()
      val tensor1Index = tensor1.storageOffset()
      val tensor2Index = tensor2.storageOffset()
      func(tensor1Data, tensor1Index, tensor2Data, tensor2Index)
      return
    }

    val tensor1Data = tensor1.storage().array()
    var tensor1Offset = tensor1.storageOffset()
    val tensor2Data = tensor2.storage().array()
    var tensor2Offset = tensor2.storageOffset()

    var adjacent = false
    if (tensor1.nDimension == 1 && tensor2.nDimension == 1 && tensor1.stride(0) == 1 &&
      tensor2.stride(0) == 1) {
      adjacent = true
    }
    if (tensor1.nDimension == 2 && tensor2.nDimension == 2) {
      if (tensor1.stride(1) == 1 && tensor2.stride(1) == 1 && tensor1.stride(0) == tensor1.size(1)
        && tensor2.stride(0) == tensor2.size(1)) {
        adjacent = true
      }

      if (tensor1.stride(0) == 1 && tensor2.stride(0) == 1 && tensor1.stride(1) == tensor1.size(0)
        && tensor2.stride(1) == tensor2.size(0)) {
        adjacent = true
      }
    }
    if (adjacent) {
      var i = 0
      while (i < tensor1.nElement()) {
        func(tensor1Data, tensor1Offset + i, tensor2Data, tensor2Offset + i)
        i += 1
      }
      return
    }

    val tensor1Stride = getStride(tensor1)
    val (largestDim1, largestSize1) = getLargestContiguousSize(tensor1)
    val counter1 = getCounter(largestDim1)
    val tensor2Stride = getStride(tensor2)
    val (largestDim2, largestSize2) = getLargestContiguousSize(tensor2)
    val counter2 = getCounter(largestDim2)

    var hasFinished = false
    var i1 = 0
    var i2 = 0
    while (!hasFinished) {
      while (i1 < largestSize1 && i2 < largestSize2) {
        func(tensor1Data, tensor1Offset + i1 * tensor1Stride, tensor2Data,
          tensor2Offset + i2 * tensor2Stride)
        i1 = i1 + 1
        i2 = i2 + 1
      }

      if (i1 == largestSize1) {
        val r = updateCounter(tensor1, counter1, tensor1Offset, largestDim1)
        hasFinished = r._1
        tensor1Offset = r._2
        i1 = 0
      }

      if (i2 == largestSize2) {
        val r = updateCounter(tensor2, counter2, tensor2Offset, largestDim2)
        hasFinished = r._1
        tensor2Offset = r._2
        i2 = 0
      }
    }
  }

  def getStride[T](tensor: Tensor[T]): Int = {
    var d = tensor.nDimension - 1
    while (d >= 0) {
      if (tensor.size(d) != 1) {
        return tensor.stride(d)
      }
      d -= 1
    }
    0
  }

  def getLargestContiguousSize[T](tensor: Tensor[T]): (Int, Int) = {
    var largestSize = 1
    var largestDim = tensor.nDimension - 1
    while (largestDim >= 0) {
      if (tensor.size(largestDim) != 1) {
        if (tensor.stride(largestDim) == largestSize) {
          largestSize = largestSize * tensor.size(largestDim)
        } else {
          return (largestDim, largestSize)
        }
      }
      largestDim -= 1
    }
    (largestDim, largestSize)
  }

  def getCounter(largestDim: Int): Array[Int] = {
    val size = largestDim + 1
    val counter = new Array[Int](size)
    var d = 0
    while (d < size) {
      counter(d) = 0
      d += 1
    }
    counter
  }

  def updateCounter[T](tensor: Tensor[T], counter: Array[Int], offset: Int, dim: Int): (Boolean, Int) = {
    if (dim == -1) {
      return (true, offset)
    }

    var _offset = offset
    var i = dim
    while (i >= 0) {
      counter(i) += 1
      _offset += tensor.stride(i)
      if (counter(i) == tensor.size(i)) {
        if (i == 0) {
          return (true, _offset)
        } else {
          _offset -= counter(i) * tensor.stride(i)
          counter(i) = 0
        }
      } else {
        return (false, _offset)
      }
      i -= 1
    }
    (false, _offset)
  }
}
