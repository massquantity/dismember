package com.mass.scalann.tensor

import scala.reflect.ClassTag

import com.mass.scalann.tensor.{DenseTensorApply => Apply}

object DenseTensorMath {

  def mul[@specialized(Float, Double) T](self: DenseTensor[T], x: Tensor[T], value: T)(implicit
      ev: TensorNumeric[T]
  ): Tensor[T] = {
    if (x != null) {
      require(self.nElement() == x.nElement())
      self.copy(x)
    }
    ev.scal(self.nElement(), value, self.storage().array(), self.storageOffset(), 1)
    self
  }

  def addmm[@specialized(Float, Double) T: ClassTag](
      r: Tensor[T],
      beta: T,
      t: Tensor[T],
      alpha: T,
      m1: Tensor[T],
      m2: Tensor[T]
  )(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(
      m1.dim() == 2 && m2.dim() == 2,
      s"matrices expected, got ${m1.dim()}, ${m2.dim()} tensors"
    )
    require(
      m1.size(1) == m2.size(0),
      s"size mismatch, m1:${m1.size().mkString("x")} m2:${m2.size().mkString("x")}"
    )
    require(t.dim() == 2, s"matrix expected, got ${t.dim()} tensor for t")
    require(
      t.size(0) == m1.size(0) && t.size(1) == m2.size(1),
      s"size mismatch. t:${t.size().mkString("x")}, " +
        s"m1:${m1.size().mkString("x")} + m2:${m2.size().mkString("x")}"
    )

    // perform matrix multiplication of m1 * m2, where sizeof m1 is m x k, m2 is k x n.
    // Since BLAS is column-major, the output size will be n x m in row-major view.
    // To get m x n size in final result, we first swap m and n.

    // override equals
    if (!r.eq(t)) {
      r.resizeAs(t).copy(t)
    }

    val m = m2.size(1)
    val n = m1.size(0)
    val k = m1.size(1)

    var transpose_m2: Char = ' '
    var lda = -1
    if (m2.stride(1) == 1 && m2.stride(0) != 0) {
      transpose_m2 = 'n'
      lda = m
    } else if (m2.stride(0) == 1 && m2.stride(1) != 0) {
      transpose_m2 = 't'
      lda = k
    }

    var transpose_m1: Char = ' '
    var ldb = -1
    if (m1.stride(1) == 1 && m1.stride(0) != 0) {
      transpose_m1 = 'n'
      ldb = k
    } else if (m1.stride(0) == 1 && m1.stride(1) != 0) {
      transpose_m1 = 't'
      ldb = n
    }
    if (lda == -1 || ldb == -1) {
      throw new IllegalArgumentException("matrix m1 and m2 must be contiguous.")
    }

    val ldc = m
    DenseTensorBLAS.gemm[T](
      transpose_m2,
      transpose_m1,
      m,
      n,
      k,
      alpha,
      m2.storage().array(),
      m2.storageOffset(),
      lda,
      m1.storage().array(),
      m1.storageOffset(),
      ldb,
      beta,
      r.storage().array(),
      r.storageOffset(),
      ldc
    )
    r
  }

  def addmv[@specialized(Float, Double) T](
      r: Tensor[T],
      beta: T,
      t: Tensor[T],
      alpha: T,
      mat: Tensor[T],
      vec: Tensor[T]
  )(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(mat.nDimension == 2 && vec.nDimension == 1)
    require(mat.size(1) == vec.size(0))
    require(t.nDimension == 1)
    require(t.size(0) == mat.size(0), s"${t.size(0)} == ${mat.size(0)}")

    if (!r.eq(t)) {
      r.resizeAs(t).copy(t)
    }

    if (mat.stride(0) == 1) {
      val lda = if (mat.size(1) == 1) {
        mat.size(0)
      } else {
        mat.stride(1)
      }
      ev.gemv(
        'N',
        mat.size(0),
        mat.size(1),
        alpha,
        mat.storage().array(),
        mat.storageOffset(),
        lda,
        vec.storage().array(),
        vec.storageOffset(),
        vec.stride(0),
        beta,
        r.storage().array(),
        r.storageOffset(),
        r.stride(0)
      )
    } else if (mat.stride(1) == 1) {
      ev.gemv(
        'T',
        mat.size(1),
        mat.size(0),
        alpha,
        mat.storage().array(),
        mat.storageOffset(),
        mat.stride(0),
        vec.storage().array(),
        vec.storageOffset(),
        vec.stride(0),
        beta,
        r.storage().array(),
        r.storageOffset(),
        r.stride(0)
      )
    } else {
      val cmat = mat.contiguous()
      ev.gemv(
        'T',
        cmat.size(1),
        cmat.size(0),
        alpha,
        cmat.storage().array(),
        cmat.storageOffset(),
        cmat.stride(0),
        vec.storage().array(),
        vec.storageOffset(),
        vec.stride(0),
        beta,
        r.storage().array(),
        r.storageOffset(),
        r.stride(0)
      )
    }
    r
  }

  def addr[@specialized(Float, Double) T](
      r: Tensor[T],
      beta: T,
      t: Tensor[T],
      alpha: T,
      vec1: Tensor[T],
      vec2: Tensor[T]
  )(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(vec1.dim() == 1 && vec2.dim() == 1)
    require(t.dim() == 2)
    require(t.size(0) == vec1.size(0) && t.size(1) == vec2.size(0))

    if (!r.eq(t)) {
      r.resizeAs(t).copy(t)
    }

    if (beta != ev.one) {
      r.mul(beta)
    }

    if (r.stride(0) == 1) {
      val lda = if (t.stride(1) == 1) {
        r.size(0)
      } else {
        r.stride(1)
      }
      ev.ger(
        vec1.size(0),
        vec2.size(0),
        alpha,
        vec1.storage().array(),
        vec1.storageOffset(),
        vec1.stride(0),
        vec2.storage().array(),
        vec2.storageOffset(),
        vec2.stride(0),
        r.storage().array(),
        r.storageOffset(),
        lda
      )
    } else if (r.stride(1) == 1) {
      ev.ger(
        vec2.size(0),
        vec1.size(0),
        alpha,
        vec2.storage().array(),
        vec2.storageOffset(),
        vec2.stride(0),
        vec1.storage().array(),
        vec1.storageOffset(),
        vec1.stride(0),
        r.storage().array(),
        r.storageOffset(),
        r.stride(0)
      )
    } else {
      val cr = r.contiguous()
      ev.ger(
        vec2.size(0),
        vec1.size(0),
        alpha,
        vec2.storage().array(),
        vec2.storageOffset(),
        vec2.stride(0),
        vec1.storage().array(),
        vec1.storageOffset(),
        vec1.stride(0),
        cr.storage().array(),
        cr.storageOffset(),
        cr.stride(0)
      )
      r.copy(cr)
    }
    r
  }

  def bmm[@specialized(Float, Double) T: ClassTag](
      result: Tensor[T],
      beta: T,
      M: Tensor[T],
      alpha: T,
      batch1: Tensor[T],
      batch2: Tensor[T]
  )(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(batch1.dim() == 3, s"expected 3D tensor, got ${batch1.dim()}D")
    require(batch2.dim() == 3, s"expected 3D tensor, got ${batch2.dim()}D")

    if (!result.eq(M)) {
      result.resizeAs(M).copy(M)
    }

    val batchSize = batch1.size(0)
    var i = 0
    while (i < batchSize) {
      val m1 = batch1.select(0, i)
      val m2 = batch2.select(0, i)
      val resultM = result.select(0, i)
      addmm(resultM, beta, resultM, alpha, m1, m2)
      i += 1
    }

    result
  }

  def cadd[@specialized(Float, Double) T](
      self: DenseTensor[T],
      x: Tensor[T],
      value: T,
      y: Tensor[T]
  )(implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(x != null && y.nElement() == x.nElement())

    if (!self.eq(x) && !self.eq(y)) {
      self.resizeAs(x).copy(x)
    }
    ev.axpy(
      y.nElement(),
      value,
      y.storage().array(),
      y.storageOffset(),
      1,
      self.storage().array(),
      self.storageOffset(),
      1
    )
    self
  }

  def cmul[@specialized T](self: DenseTensor[T], x: DenseTensor[T], y: DenseTensor[T])(implicit
      ev: TensorNumeric[T]
  ): Tensor[T] = {
    ev.vMul(
      self.nElement(),
      x.storage().array(),
      x.storageOffset(),
      y.storage().array(),
      y.storageOffset(),
      self.storage().array(),
      self.storageOffset()
    )
    self
  }

  def cdiv[@specialized(Float, Double) T](self: DenseTensor[T], x: Tensor[T], y: Tensor[T])(implicit
      ev: TensorNumeric[T]
  ): Tensor[T] = {
    ev.vDiv(
      self.nElement(),
      x.storage().array(),
      x.storageOffset(),
      y.storage().array(),
      y.storageOffset(),
      self.storage().array(),
      self.storageOffset()
    )
    self
  }

  def log[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], x: Tensor[T])(implicit
      ev: TensorNumeric[T]
  ): Tensor[T] = {
    require(self.nElement() == x.nElement())
    ev.vLn(
      self.nElement(),
      x.storage().array(),
      x.storageOffset(),
      self.storage().array(),
      self.storageOffset()
    )
    self
  }

  def pow[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], x: Tensor[T], n: T)(
      implicit ev: TensorNumeric[T]
  ): Tensor[T] = {
    require(self.nElement() == x.nElement())
    ev.vPowx(
      self.nElement(),
      x.storage().array(),
      x.storageOffset(),
      n,
      self.storage().array(),
      self.storageOffset()
    )
    self
  }

  def exp[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], x: Tensor[T])(implicit
      ev: TensorNumeric[T]
  ): Tensor[T] = {
    if (self.nElement() != x.nElement()) {
      self.resizeAs(x)
    }
    ev.vExp(
      self.nElement(),
      x.storage().array(),
      x.storageOffset(),
      self.storage().array(),
      self.storageOffset()
    )
    self
  }

  def sqrt[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T], x: Tensor[T])(implicit
      ev: TensorNumeric[T]
  ): Tensor[T] = {
    ev.vSqrt(
      self.nElement(),
      x.storage().array(),
      x.storageOffset(),
      self.storage().array(),
      self.storageOffset()
    )
    self
  }

  def sumAll[@specialized(Float, Double) T](
      self: DenseTensor[T]
  )(implicit ev: TensorNumeric[T]): T = {
    var sum = ev.fromType[Int](0)
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit = {
        sum = ev.plus(data(index), sum)
      }
    }
    Apply.apply1[T](self, func)
    sum
  }

  def sum[@specialized T: ClassTag](self: DenseTensor[T], x: Tensor[T], _dim: Int)(implicit
      ev: TensorNumeric[T]
  ): Tensor[T] = {
    require(_dim >= 0 && _dim < x.nDimension, s"dimension ${_dim} out of arange")
    val result = if (self == null) new DenseTensor[T]() else self
    val sizes = x.size().clone() // copy size, because will modify later
    sizes(_dim) = 1
    /*
    val sizes = new Array[Int](x.size().length - 1)
    var i = 0
    var j = 0
    while (i < x.size().length) {
      if (i != 1) {
        sizes(j) = x.size()(i)
        j += 1
      }
      i += 1
    } */

    result.resize(sizes)
    DenseTensorDimApply.dimApply2[T](
      result,
      x,
      _dim,
      (rData, rOffset, rStride, rSize, tData, tOffset, tStride, tSize) => {
        rData(rOffset) = ev.sum(tSize, tData, tOffset, tStride)
      }
    )
    result
  }

  def maxAll[@specialized(Float, Double) T](
      self: DenseTensor[T]
  )(implicit ev: TensorNumeric[T]): T = {
    var max = ev.fromType[Int](0)
    var first = true
    val func = new TensorFunc2[T] {
      override def apply(data: Array[T], index: Int): Unit = {
        if (first) {
          first = false
          max = data(index)
        } else if (ev.isGreater(data(index), max)) {
          max = data(index)
        }
      }
    }
    Apply.apply1[T](self, func)
    max
  }
}
