package com.mass.sparkdl.tensor

import java.util.{Arrays => JArrays}

import com.intel.analytics.bigdl.mkl.MKL

trait TensorNumeric[@specialized(Float, Double) T] extends Serializable {
  def fromType[K](k: K)(implicit c: ConvertableFrom[K]): T

  def toType[K](t: T)(implicit c: ConvertableTo[K]): K

  def one: T = fromType[Int](1)

  def zero: T = fromType[Int](0)

  def plus(x: T, y: T): T

  def minus(x: T, y: T): T

  def times(x: T, y: T): T

  def divide(x: T, y: T): T

  def max(x: T, y: T): T

  def exp(x: T): T

  def log(x: T): T

  def inv(v: T): T

  def abs(x: T): T

  def abs(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit = { }

  def getType: TensorDataType

  def arraycopy(src: Array[T], srcPos: Int, dest: Array[T],
    destPos: Int, length: Int): Unit = { }

  def axpy(n: Int, da: T, dx: Array[T], _dx_offset: Int, incx: Int, dy: Array[T],
    _dy_offset: Int, incy: Int): Unit = { }

  def gemm(transa: Char, transb: Char, m: Int, n: Int, k: Int, alpha: T, a: Array[T],
    aOffset: Int, lda: Int, b: Array[T], bOffset: Int, ldb: Int, beta: T, c: Array[T],
    cOffset: Int, ldc: Int): Unit = { }

  def gemv(trans: Char, m: Int, n: Int, alpha: T, a: Array[T], aoffset: Int, lda: Int,
    x: Array[T], xOffset: Int, incx: Int, beta: T, y: Array[T], yOffset: Int, incy: Int)
    : Unit = { }

  def ger(m: Int, n: Int, alpha: T, x: Array[T], _x_offset: Int, incx: Int, y: Array[T],
    _y_offset: Int, incy: Int, a: Array[T], _a_offset: Int, lda: Int): Unit = { }

  def scal(n: Int, sa: T, sx: Array[T], offset: Int, incx: Int): Unit = { }

  def vAdd(n: Int, a: Array[T], aOffset: Int, b: Array[T], bOffset: Int, y: Array[T],
    yOffset: Int): Unit = { }

  def vSub(n: Int, a: Array[T], aOffset: Int, b: Array[T], bOffset: Int, y: Array[T],
    yOffset: Int): Unit = { }

  def vMul(n: Int, a: Array[T], aOffset: Int, b: Array[T], bOffset: Int, y: Array[T],
    yOffset: Int): Unit = { }

  def vDiv(n: Int, a: Array[T], aOffset: Int, b: Array[T], bOffset: Int, y: Array[T],
    yOffset: Int): Unit = { }

  def add(n: Int, a: Array[T], offset: Int, v: T, stride: Int): Unit = { }

  def sub(n: Int, a: Array[T], offset: Int, v: T, stride: Int): Unit = { }

  def addcdiv(value: T, n: Int, self: Array[T], selfOffset: Int, a: Array[T], aOffset: Int,
    b: Array[T], bOffset: Int): Unit = { }

  def vLn(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit = { }

  def vPowx(n: Int, a: Array[T], aOffset: Int, b: T, y: Array[T], yOffset: Int): Unit = { }

  def vExp(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit = { }

  def vSqrt(n: Int, a: Array[T], aOffset: Int, y: Array[T], yOffset: Int): Unit = { }

  def dot(n: Int, dx: Array[T], _dx_offset: Int, incx: Int, dy: Array[T], _dy_offset: Int,
    incy: Int): T = throw new UnsupportedOperationException()

  def isGreater(x: T, y: T): Boolean

  def isGreaterEq(x: T, y: T): Boolean

  def negative(x: T): T

  def clip(a: T, lower: T, upper: T): T = this.fromType(0)

  def sum(n: Int, a: Array[T], aOffset: Int, stride: Int): T
}

object TensorNumeric {

  implicit object NumericFloat extends TensorNumeric[Float] {

    override def fromType[@specialized(Float, Double, Int) K](k: K)(
      implicit c: ConvertableFrom[K]): Float = c.toFloat(k)

    override def toType[@specialized(Float, Double, Int) K](t: Float)(
      implicit c: ConvertableTo[K]): K = c.fromFloat(t)

    override def plus(x: Float, y: Float): Float = x + y

    override def minus(x: Float, y: Float): Float = x - y

    override def times(x: Float, y: Float): Float = x * y

    override def divide(x: Float, y: Float): Float = x / y

    override def max(x: Float, y: Float): Float = java.lang.Math.max(x, y)

    override def exp(x: Float): Float = java.lang.Math.exp(x).toFloat

    override def log(x: Float): Float = java.lang.Math.log(x).toFloat

    override def inv(v: Float): Float = 1 / v

    override def abs(x: Float): Float = java.lang.Math.abs(x)

    override def abs(n: Int, a: Array[Float], aOffset: Int, y: Array[Float], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vsAbs(n, a, aOffset, y, yOffset)
    }

    override def getType: TensorDataType = FloatType

    override def arraycopy(src: Array[Float], srcPos: Int, dest: Array[Float],
        destPos: Int, length: Int): Unit = {
      System.arraycopy(src, srcPos, dest, destPos, length)
    }

    override def axpy(n: Int, da: Float, dx: Array[Float], _dx_offset: Int, incx: Int,
        dy: Array[Float], _dy_offset: Int, incy: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vsaxpy(n, da, dx, _dx_offset, incx, dy, _dy_offset, incy)
    }

    override def gemm(transa: Char, transb: Char, m: Int, n: Int, k: Int, alpha: Float,
        a: Array[Float], aOffset: Int, lda: Int, b: Array[Float], bOffset: Int, ldb: Int,
        beta: Float, c: Array[Float], cOffset: Int, ldc: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vsgemm(transa, transb, m, n, k, alpha, a, aOffset, lda, b, bOffset,
        ldb, beta, c, cOffset, ldc)
    }

    override def gemv(trans: Char, m: Int, n: Int, alpha: Float, a: Array[Float],
        aoffset: Int, lda: Int, x: Array[Float], xOffset: Int, incx: Int, beta: Float,
        y: Array[Float], yOffset: Int, incy: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vsgemv(trans, m, n, alpha, a, aoffset, lda, x, xOffset,
        incx, beta, y, yOffset, incy)
    }

    override def ger(m: Int, n: Int, alpha: Float, x: Array[Float], _x_offset: Int,
        incx: Int, y: Array[Float], _y_offset: Int, incy: Int, a: Array[Float],
        _a_offset: Int, lda: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vsger(m, n, alpha, x, _x_offset, incx, y, _y_offset,
        incy, a, _a_offset, lda)
    }

    override def scal(n: Int, sa: Float, sx: Array[Float], offset: Int, incx: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vsscal(n, sa, sx, offset, incx)
    }

    override def vSub(n: Int, a: Array[Float], aOffset: Int, b: Array[Float], bOffset: Int,
        y: Array[Float], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vsSub(n, a, aOffset, b, bOffset, y, yOffset)
    }

    override def vMul(n: Int, a: Array[Float], aOffset: Int, b: Array[Float], bOffset: Int,
        y: Array[Float], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vsMul(n, a, aOffset, b, bOffset, y, yOffset)
    }

    override def vDiv(n: Int, a: Array[Float], aOffset: Int, b: Array[Float], bOffset: Int,
        y: Array[Float], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vsDiv(n, a, aOffset, b, bOffset, y, yOffset)
    }

    override def add(n: Int, a: Array[Float], offset: Int, v: Float, stride: Int): Unit = {
      var i = 0
      while (i < n) {
        a(offset + i * stride) += v
        i += 1
      }
    }

    override def sub(n: Int, a: Array[Float], offset: Int, v: Float, stride: Int): Unit = {
      var i = 0
      while (i < n) {
        a(offset + i * stride) -= v
        i += 1
      }
    }

    override def vAdd(n: Int, a: Array[Float], aOffset: Int, b: Array[Float], bOffset: Int,
        y: Array[Float], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vsAdd(n, a, aOffset, b, bOffset, y, yOffset)
    }

    override def addcdiv(value: Float, n: Int, self: Array[Float], selfOffset: Int,
        a: Array[Float], aOffset: Int, b: Array[Float], bOffset: Int): Unit = {
      val v = value.asInstanceOf[Float]
      var i = 0
      while (i < n) {
        self(i + selfOffset) += a(aOffset + i) / b(bOffset + i) * v
        i += 1
      }
    }

    override def vLn(n: Int, a: Array[Float], aOffset: Int, y: Array[Float], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vsLn(n, a, aOffset, y, yOffset)
    }

    override def vPowx(n: Int, a: Array[Float], aOffset: Int, b: Float, y: Array[Float],
        yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vsPowx(n, a, aOffset, b, y, yOffset)
    }

    override def vExp(n: Int, a: Array[Float], aOffset: Int, y: Array[Float], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vsExp(n, a, aOffset, y, yOffset)
    }

    override def vSqrt(n: Int, a: Array[Float], aOffset: Int, y: Array[Float], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vsSqrt(n, a, aOffset, y, yOffset)
    }

    override def dot(n: Int, dx: Array[Float], _dx_offset: Int, incx: Int, dy: Array[Float],
        _dy_offset: Int, incy: Int): Float = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vsdot(n, dx, _dx_offset, incx, dy, _dy_offset, incy)
    }

    override def isGreater(x: Float, y: Float): Boolean = x > y

    override def isGreaterEq(x: Float, y: Float): Boolean = x >= y

    override def negative(x: Float): Float = -x

    override def clip(a: Float, lower: Float, upper: Float): Float = {
      require(lower <= upper, "lower bound must be less or equal than upper bound")
      math.min(math.max(a, lower), upper)
    }

    override def sum(n: Int, a: Array[Float], aOffset: Int, stride: Int): Float = {
      var i = 0
      var r = 0.0f
      while (i < n) {
        r += a(aOffset + i * stride)
        i += 1
      }
      r
    }
  }

  implicit object NumericDouble extends TensorNumeric[Double] {

    override def fromType[@specialized(Float, Double, Int) K](k: K)(
      implicit c: ConvertableFrom[K]): Double = c.toDouble(k)

    override def toType[@specialized(Float, Double, Int)K](t: Double)(
      implicit c: ConvertableTo[K]): K = c.fromDouble(t)

    override def plus(x: Double, y: Double): Double = x + y

    override def minus(x: Double, y: Double): Double = x - y

    override def times(x: Double, y: Double): Double = x * y

    override def divide(x: Double, y: Double): Double = x / y

    override def max(x: Double, y: Double): Double = java.lang.Math.max(x, y)

    override def exp(x: Double): Double = java.lang.Math.exp(x)

    override def log(x: Double): Double = java.lang.Math.log(x)

    override def inv(v: Double): Double = 1 / v

    override def abs(x: Double): Double = java.lang.Math.abs(x)

    override def abs(n: Int, a: Array[Double], aOffset: Int, y: Array[Double], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vdAbs(n, a, aOffset, y, yOffset)
    }

    override def getType: TensorDataType = DoubleType

    override def arraycopy(src: Array[Double], srcPos: Int, dest: Array[Double],
        destPos: Int, length: Int): Unit = {
      System.arraycopy(src, srcPos, dest, destPos, length)
    }

    override def axpy(n: Int, da: Double, dx: Array[Double], _dx_offset: Int, incx: Int,
        dy: Array[Double], _dy_offset: Int, incy: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vdaxpy(n, da, dx, _dx_offset, incx, dy, _dy_offset, incy)
    }

    override def gemm(transa: Char, transb: Char, m: Int, n: Int, k: Int, alpha: Double,
        a: Array[Double], aOffset: Int, lda: Int, b: Array[Double], bOffset: Int, ldb: Int,
        beta: Double, c: Array[Double], cOffset: Int, ldc: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vdgemm(transa, transb, m, n, k, alpha, a, aOffset, lda, b,
        bOffset, ldb, beta, c, cOffset, ldc)
    }

    override def gemv(trans: Char, m: Int, n: Int, alpha: Double, a: Array[Double],
        aoffset: Int, lda: Int, x: Array[Double], xOffset: Int, incx: Int, beta: Double,
        y: Array[Double], yOffset: Int, incy: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vdgemv(trans, m, n, alpha, a, aoffset, lda, x, xOffset,
        incx, beta, y, yOffset, incy)
    }

    override def ger(m: Int, n: Int, alpha: Double, x: Array[Double], _x_offset: Int,
        incx: Int, y: Array[Double], _y_offset: Int, incy: Int, a: Array[Double],
        _a_offset: Int, lda: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vdger(m, n, alpha, x, _x_offset, incx, y, _y_offset,
        incy, a, _a_offset, lda)
    }

    override def scal(n: Int, sa: Double, sx: Array[Double], offset: Int, incx: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vdscal(n, sa, sx, offset, incx)
    }

    override def vSub(n: Int, a: Array[Double], aOffset: Int, b: Array[Double], bOffset: Int,
        y: Array[Double], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vdSub(n, a, aOffset, b, bOffset, y, yOffset)
    }

    override def vMul(n: Int, a: Array[Double], aOffset: Int, b: Array[Double], bOffset: Int,
        y: Array[Double], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vdMul(n, a, aOffset, b, bOffset, y, yOffset)
    }

    override def vDiv(n: Int, a: Array[Double], aOffset: Int, b: Array[Double], bOffset: Int,
        y: Array[Double], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vdDiv(n, a, aOffset, b, bOffset, y, yOffset)
    }

    override def add(n: Int, a: Array[Double], offset: Int, v: Double, stride: Int): Unit = {
      var i = 0
      while (i < n) {
        a(offset + i * stride) += v
        i += 1
      }
    }

    override def sub(n: Int, a: Array[Double], offset: Int, v: Double, stride: Int): Unit = {
      var i = 0
      while (i < n) {
        a(offset + i * stride) -= v
        i += 1
      }
    }

    override def vAdd(n: Int, a: Array[Double], aOffset: Int, b: Array[Double], bOffset: Int,
        y: Array[Double], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vdAdd(n, a, aOffset, b, bOffset, y, yOffset)
    }

    override def addcdiv(value: Double, n: Int, self: Array[Double], selfOffset: Int,
        a: Array[Double], aOffset: Int, b: Array[Double], bOffset: Int): Unit = {
      val v = value.asInstanceOf[Double]
      var i = 0
      while (i < n) {
        self(i + selfOffset) += a(aOffset + i) / b(bOffset + i) * v
        i += 1
      }
    }

    override def vLn(n: Int, a: Array[Double], aOffset: Int, y: Array[Double], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vdLn(n, a, aOffset, y, yOffset)
    }

    override def vPowx(n: Int, a: Array[Double], aOffset: Int, b: Double, y: Array[Double],
        yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vdPowx(n, a, aOffset, b, y, yOffset)
    }

    override def vExp(n: Int, a: Array[Double], aOffset: Int, y: Array[Double], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vdExp(n, a, aOffset, y, yOffset)
    }

    override def vSqrt(n: Int, a: Array[Double], aOffset: Int, y: Array[Double], yOffset: Int): Unit = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vdSqrt(n, a, aOffset, y, yOffset)
    }

    override def dot(n: Int, dx: Array[Double], _dx_offset: Int, incx: Int, dy: Array[Double],
        _dy_offset: Int, incy: Int): Double = {
      require(MKL.isMKLLoaded, "mkl hasn't been loaded")
      MKL.vddot(n, dx, _dx_offset, incx, dy, _dy_offset, incy)
    }

    override def isGreater(x: Double, y: Double): Boolean = x > y

    override def isGreaterEq(x: Double, y: Double): Boolean = x >= y

    override def negative(x: Double): Double = -x

    override def clip(a: Double, lower: Double, upper: Double): Double = {
      require(lower <= upper, "lower bound must be less or equal than upper bound")
      math.min(math.max(a, lower), upper)
    }

    override def sum(n: Int, a: Array[Double], aOffset: Int, stride: Int): Double = {
      var i = 0
      var r = 0.0
      while (i < n) {
        r += a(aOffset + i * stride)
        i += 1
      }
      r
    }
  }

  implicit object NumericInt extends TensorNumeric[Int] {

    override def fromType[K](k: K)(implicit c: ConvertableFrom[K]): Int = c.toInt(k)

    override def toType[K](t: Int)(implicit c: ConvertableTo[K]): K = c.fromInt(t)

    override def plus(x: Int, y: Int): Int = x + y

    override def minus(x: Int, y: Int): Int = x - y

    override def times(x: Int, y: Int): Int = x * y

    override def divide(x: Int, y: Int): Int = x / y

    override def max(x: Int, y: Int): Int = java.lang.Math.max(x, y)

    override def exp(x: Int): Int = java.lang.Math.exp(x).toInt

    override def log(x: Int): Int = java.lang.Math.log(x).toInt

    override def inv(v: Int): Int = 1 / v

    override def abs(x: Int): Int = java.lang.Math.abs(x)

    override def getType: TensorDataType = IntType

    override def axpy(n: Int, da: Int, dx: Array[Int], _dx_offset: Int, incx: Int,
        dy: Array[Int], _dy_offset: Int, incy: Int): Unit = {
      var i = 0
      while (i < n) {
        dy(i + _dy_offset) = dx(_dx_offset + i) + dy(_dy_offset + i)
        i += 1
      }
    }

    override def sub(n: Int, a: Array[Int], offset: Int, v: Int, stride: Int): Unit = {
      var i = 0
      while(i < n) {
        a(i * stride + offset) -= v
        i += 1
      }
    }

    override def vMul(n: Int, a: Array[Int], aOffset: Int, b: Array[Int], bOffset: Int,
        y: Array[Int], yOffset: Int): Unit = {
      var i = 0
      while(i < n) {
        y(i + yOffset) = a(i + aOffset) * b(i + bOffset)
        i += 1
      }
    }

    override def vDiv(n: Int, a: Array[Int], aOffset: Int, b: Array[Int], bOffset: Int,
      y: Array[Int], yOffset: Int): Unit = {
      var i = 0
      while(i < n) {
        y(i + yOffset) = a(i + aOffset) / b(i + bOffset)
        i += 1
      }
    }

    override def isGreater(x: Int, y: Int): Boolean = x > y

    override def isGreaterEq(x: Int, y: Int): Boolean = x >= y

    override def negative(x: Int): Int = -x

    override def sum(n: Int, a: Array[Int], aOffset: Int, stride: Int): Int = {
      var i = 0
      var r = 0
      while (i < n) {
        r += a(aOffset + i * stride)
        i += 1
      }
      r
    }
  }

  implicit object NumericLong extends TensorNumeric[Long] {

    override def fromType[K](k: K)(implicit c: ConvertableFrom[K]): Long = c.toLong(k)

    override def toType[K](t: Long)(implicit c: ConvertableTo[K]): K = c.fromLong(t)

    override def plus(x: Long, y: Long): Long = x + y

    override def minus(x: Long, y: Long): Long = x - y

    override def times(x: Long, y: Long): Long = x * y

    override def divide(x: Long, y: Long): Long = x / y

    override def max(x: Long, y: Long): Long = java.lang.Math.max(x, y)

    override def exp(x: Long): Long = java.lang.Math.exp(x).toLong

    override def log(x: Long): Long = java.lang.Math.log(x).toLong

    override def inv(v: Long): Long = 1 / v

    override def abs(x: Long): Long = java.lang.Math.abs(x)

    override def getType: TensorDataType = LongType

    override def axpy(n: Int, da: Long, dx: Array[Long], _dx_offset: Int, incx: Int,
        dy: Array[Long], _dy_offset: Int, incy: Int): Unit = {
      var i = 0
      while (i < n) {
        dy(i + _dy_offset) = dx(_dx_offset + i) + dy(_dy_offset + i)
        i += 1
      }
    }

    override def isGreater(x: Long, y: Long): Boolean = x > y

    override def isGreaterEq(x: Long, y: Long): Boolean = x >= y

    override def negative(x: Long): Long = -x

    override def sum(n: Int, a: Array[Long], aOffset: Int, stride: Int): Long = {
      var i = 0
      var r = 0L
      while (i < n) {
        r += a(aOffset + i * stride)
        i += 1
      }
      r
    }
  }
}
