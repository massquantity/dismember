package com.mass.sparkdl.tensor



object DenseTensorBLAS {
  var time = 0L

  /**
   * The gemm routines compute a scalar-matrix-matrix product and
   * add the result to a scalar-matrix product, with general matrices.
   * C := alpha*op(A)*op(B) + beta*C,
   * where:
   * op(X) is one of op(X) = X, or op(X) = XT,
   * alpha and beta are scalars,
   * A, B and C are matrices:
   * op(A) is an m-by-k matrix,
   * op(B) is a k-by-n matrix,
   * C is an m-by-n matrix.
   */
  def gemm[@specialized(Float, Double) T](
      transa: Char,
      transb: Char,
      m: Int,
      n: Int,
      k: Int,
      alpha: T,
      a: Array[T],
      aOffset: Int,
      lda: Int,
      b: Array[T],
      bOffset: Int,
      ldb: Int,
      beta: T,
      c: Array[T],
      cOffset: Int,
      ldc: Int)(implicit ev: TensorNumeric[T]): Unit = {

    val _transa = (transa == 't' || transa == 'T')
    val _transb = (transb == 't' || transb == 'T')

    val _ldc = if (n == 1) m else ldc

    var _lda = lda
    if (_transa) {
      if (m == 1) {
        _lda = k
      }
    } else {
      if (k == 1) {
        _lda = m
      }
    }

    var _ldb = ldb
    if (_transb) {
      if (k == 1) {
        _ldb = n
      }
    } else {
      if (n == 1) {
        _ldb = k
      }
    }

    val start = System.nanoTime()
    ev.gemm(transa, transb, m, n, k, alpha, a, aOffset, _lda, b, bOffset, _ldb,
      beta, c, cOffset, _ldc)
    time += (System.nanoTime() - start)
  }

  /**
   * The gemv routines perform a matrix-vector operation defined as
   * y := alpha*A*x + beta*y,
   * or
   * y := alpha*A'*x + beta*y,
   * where:
   * alpha and beta are scalars,
   * x and y are vectors,
   * A is an m-by-n matrix.
   */
  def gemv[@specialized(Float, Double) T](
      trans: Char,
      m: Int,
      n: Int,
      alpha: T,
      a: Array[T],
      aOffset: Int,
      lda: Int,
      x: Array[T],
      xOffset: Int,
      incx: Int,
      beta: T,
      y: Array[T],
      yOffset: Int,
      incy: Int)(implicit ev: TensorNumeric[T]): Unit = {

    val start = System.nanoTime()
    ev.gemv(trans, m, n, alpha, a, aOffset, lda, x, xOffset, incx, beta, y,
      yOffset, incy)
    time += (System.nanoTime() - start)
  }

  /**
   * The ger routines perform a matrix-vector operation defined as
   * A := alpha*x*y'+ A,
   * where:
   * alpha is a scalar,
   * x is an m-element vector,
   * y is an n-element vector,
   * A is an m-by-n general matrix.
   */
  def ger[@specialized(Float, Double) T](
      m: Int,
      n: Int,
      alpha: T,
      x: Array[T],
      xOffset: Int,
      incx: Int,
      y: Array[T],
      yOffset: Int,
      incy: Int,
      a: Array[T],
      aOffset: Int,
      lda: Int)(implicit ev: TensorNumeric[T]): Unit = {

    val start = System.nanoTime()
    ev.ger(m, n, alpha, x, xOffset, incx, y, yOffset, incy, a, aOffset, lda)
    time += (System.nanoTime() - start)
  }
}
