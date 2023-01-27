package com.mass.scalann.tensor

object DenseTensorDimApply {

  private[tensor] def dimApply2[@specialized(Float, Double) T](
      tensor1: DenseTensor[T],
      tensor2: Tensor[T],
      _dim: Int,
      func: (Array[T], Int, Int, Int, Array[T], Int, Int, Int) => Unit
  ): Unit = {

    require(_dim >= 0 && _dim < tensor1.nDimension, "invalid dimension")
    require(tensor1.nDimension == tensor2.nDimension, "inconsistent tensor sizes")

    val counter = new Array[Int](tensor1.nDimension)
    val _data1 = tensor1.storage().array()
    var _offset1 = tensor1.storageOffset()
    val stride1 = tensor1.stride(_dim)
    val size1 = tensor1.size(_dim)

    val _data2 = tensor2.storage().array()
    var _offset2 = tensor2.storageOffset()
    val stride2 = tensor2.stride(_dim)
    val size2 = tensor2.size(_dim)

    var hasFinished = false
    while (!hasFinished) {
      func(_data1, _offset1, stride1, size1, _data2, _offset2, stride2, size2)

      if (tensor1.nDimension == 1) {
        hasFinished = true
      } else {
        var i = 0
        var break = false
        while (i < tensor1.nDimension && !break) {
          if (i == _dim) {
            if (i == tensor1.nDimension - 1) {
              hasFinished = true
              break = true
            }
          } else {
            counter(i) += 1
            _offset1 += tensor1.stride(i)
            _offset2 += tensor2.stride(i)

            if (counter(i) == tensor1.size(i)) {
              if (i == tensor1.nDimension - 1) {
                break = true
                hasFinished = true
              } else {
                _offset1 -= counter(i) * tensor1.stride(i)
                _offset2 -= counter(i) * tensor2.stride(i)
                counter(i) = 0
              }
            } else {
              break = true
            }
          }
          i += 1
        }
      }
    }
  }
}
