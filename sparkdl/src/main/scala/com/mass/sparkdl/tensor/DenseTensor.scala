package com.mass.sparkdl.tensor

import scala.reflect.ClassTag

import org.apache.commons.math3.random.RandomDataGenerator
import org.apache.spark.mllib.linalg.{Vector, Matrix, DenseVector, DenseMatrix}

import com.mass.sparkdl.utils.File

private[tensor] class DenseTensor[@specialized T: ClassTag](
    private[tensor] var _storage: ArrayStorage[T],
    private[tensor] var _storageOffset: Int,
    private[tensor] var _size: Array[Int],
    private[tensor] var _stride: Array[Int],
    var nDimension: Int)(implicit ev: TensorNumeric[T]) extends Tensor[T] {

  private[tensor] def this()(implicit ev: TensorNumeric[T]) = this(null, 0, null, null, 0)

  private[tensor] def this(d1: Int)(implicit ev: TensorNumeric[T]) =
    this(new ArrayStorage[T](new Array[T](d1)), 0, Array(d1), Array(1), 1)

  private[tensor] def this(d1: Int, d2: Int)(implicit ev: TensorNumeric[T]) =
    this(new ArrayStorage[T](new Array[T](d1 * d2)), 0, Array(d1, d2),
      Array(d2, 1), 2)

  private[tensor] def this(d1: Int, d2: Int, d3: Int)(implicit ev: TensorNumeric[T]) =
    this(new ArrayStorage[T](new Array[T](d1 * d2 * d3)), 0, Array(d1, d2, d3),
      Array(d3 * d2, d3, 1), 3)

  private[tensor] def this(dims: Int*)(implicit ev: TensorNumeric[T]) =
    this(new ArrayStorage[T](new Array[T](dims.product)), 0, dims.toArray,
      DenseTensor.size2Stride(dims.toArray), dims.length)

  private[tensor] def this(storage: ArrayStorage[T])(implicit ev: TensorNumeric[T]) = {
    this(null, 0, null, null, 0)
    val _storageOffset = 0
    val _size = Array(storage.length)
    val _stride = Array(1)
    DenseTensor.newWithStorage(this, storage, _storageOffset, _size, _stride, ev)
  }

  private[tensor] def this(storage: ArrayStorage[T], storageOffset: Int, size: Array[Int] = null,
      stride: Array[Int] = null)(implicit ev: TensorNumeric[T]) = {
    this(null, 0, null, null, 0)
    if (storage != null) {
      val _storageOffset = storageOffset
      val _size = if (size == null) Array(storage.length) else size
      val _stride = if (size == null) null else stride
      DenseTensor.newWithStorage(this, storage, _storageOffset, _size, _stride, ev)
    }
  }

  private[tensor] def this(other: Tensor[T])(implicit ev: TensorNumeric[T]) = {
    this(null, 0, null, null, 0)
    require(other.isInstanceOf[DenseTensor[_]], "Only support dense tensor in this operation")
    val _storage = other.storage().asInstanceOf[ArrayStorage[T]]
    val _storageOffset = other.storageOffset()
    val _size = other.size()
    val _stride = other.stride()
    DenseTensor.newWithStorage(this, _storage, _storageOffset, _size, _stride, ev)
  }

  override def isEmpty: Boolean = storage() == null || storage().length == 0

  override def isScalar: Boolean = !isEmpty && nDimension == 0

  override def storage(): Storage[T] = _storage

  override def storageOffset(): Int = _storageOffset

  override def dim(): Int = nDimension

  override def nElement(): Int = {
    if (isEmpty) {
      0
    } else {
      var n = 1
      var d = 0
      while (d < this.nDimension) {
        n = n * this._size(d)
        d += 1
      }
      n
    }
  }

  override def size(): Array[Int] = _size

  override def size(dim: Int): Int = {
    require(dim >= 0 && dim < nDimension, "dimension out of arange")
    _size(dim)
  }

  override def stride(): Array[Int] = _stride

  override def stride(dim: Int): Int = {
    require(dim >= 0 && dim < nDimension, "dimension out of arange")
    _stride(dim)
  }

  override def fill(v: T): Tensor[T] = {
    if (this.storage() == null) return this

    if (this.isContiguous) {
      this.storage().fill(v, this.storageOffset(), this.nElement())
    } else {
      val func = new TensorFunc2[T] {
        override def apply(data: Array[T], index: Int): Unit = {
          data(index) = v
        }
      }
      DenseTensorApply.apply1[T](this, func)
    }
    this
  }

  override def zero(): Tensor[T] = this.fill(ev.fromType(0))

  override def randn(mean: Double, stdv: Double,
      seed: Long = System.nanoTime()): Tensor[T] = {
    val generator = new RandomDataGenerator()
    generator.reSeed(seed)
    var i = 0
    val total = this.nElement()
    val data = this.storage().array()
    val offset = this.storageOffset()
    while (i < total) {
      data(offset + i) = ev.fromType(generator.nextGaussian(mean, stdv))
      i += 1
    }
    this
  }

  override def rand(lowerBound: Double, upperBound: Double,
      seed: Long = System.nanoTime()): Tensor[T] = {
    val generator = new RandomDataGenerator()
    generator.reSeed(seed)
    var i = 0
    val total = this.nElement()
    val data = this.storage().array()
    val offset = this.storageOffset()
    while (i < total) {
      data(offset + i) = ev.fromType(generator.nextUniform(lowerBound, upperBound))
      i += 1
    }
    this
  }

  override def randInt(lowerBound: Int, upperBound: Int,
      seed: Long = System.nanoTime()): Tensor[T] = {
    val generator = new RandomDataGenerator()
    generator.reSeed(seed)
    var i = 0
    val total = this.nElement()
    val data = this.storage().array()
    val offset = this.storageOffset()
    while (i < total) {
      data(offset + i) = ev.fromType(generator.nextInt(lowerBound, upperBound))
      i += 1
    }
    this
  }

  override def transpose(dim1: Int, dim2: Int): Tensor[T] = {
    require(dim1 >= 0 && dim2 < nDimension, "dim1 out of arange")
    require(dim2 >= 0 && dim2 < nDimension, "dim2 out of arange")
    val newTensor = DenseTensor.newWithTensor(this)
    if (dim1 != dim2) {
      var tmp = newTensor._stride(dim1)
      newTensor._stride(dim1) = newTensor._stride(dim2)
      newTensor._stride(dim2) = tmp
      tmp = newTensor._size(dim1)
      newTensor._size(dim1) = newTensor._size(dim2)
      newTensor._size(dim2) = tmp
    }
    newTensor
  }

  override def t(): Tensor[T] = {
    require(nDimension == 2, "t() only supports 2D tensor")
    transpose(0, 1)
  }

  override def apply(index: Int): Tensor[T] = {
    var _index = index
    val dim_size = this._size(0)
    if (_index < 0) _index = dim_size + _index
    require(_index >= 0 && _index < dim_size, "index out of arange")
    val result = DenseTensor.newWithTensor(this)
    DenseTensor.select(result, null, 0, _index)
    result
  }

  override def apply(indexes: Array[Int]): T = {
    require(indexes.length == this.nDimension, "invalid size")
    var offset = this._storageOffset
    var d = 0
    while (d < indexes.length) {
      offset += getOffset(indexes(d), d)
      d += 1
    }
    this._storage(offset)
  }

  override def value(): T = {
    require(1 == this.nElement(), s"invalid size: 1 == ${this.nElement()}")
    val offset = this._storageOffset
    this._storage(offset)
  }

  override def valueAt(d1: Int): T = {
    require(1 == this.nDimension, s"invalid size: 1 == ${this.nDimension}")
    var offset = this._storageOffset
    offset += getOffset(d1, 0)
    this._storage(offset)
  }

  override def valueAt(d1: Int, d2: Int): T = {
    require(2 == this.nDimension, "invalid size")
    var offset = this._storageOffset
    offset += getOffset(d1, 0)
    offset += getOffset(d2, 1)
    this._storage(offset)
  }

  override def setValue(value: T): this.type = {
    require(0 == this.nDimension, "invalid size, you can only call this on a scalar")
    val offset = this._storageOffset
    this._storage(offset) = value
    this
  }

  override def setValue(d1: Int, value: T): this.type = {
    require(1 == this.nDimension, "invalid size")
    var offset = this._storageOffset
    offset += getOffset(d1, 0)
    this._storage(offset) = value
    this
  }

  override def setValue(d1: Int, d2: Int, value: T): this.type = {
    require(2 == this.nDimension, "invalid size")
    var offset = this._storageOffset
    offset += getOffset(d1, 0)
    offset += getOffset(d2, 1)
    this._storage(offset) = value
    this
  }

  override def update(index: Int, value: T): Unit = {
    require(this.nDimension > 0, "empty tensor")
    var _index = index
    if (_index < 0) _index = this._size(0) + _index
    require(_index >= 0 && _index < this._size(0), "out of arange")
    if (this.nDimension == 1) {
      this._storage(_storageOffset + _index * _stride(0)) = value
    } else {
      val tensor = DenseTensor.newWithTensor(this)
      DenseTensor.narrow(tensor, null, 0, _index, 1)
      tensor.fill(value)
    }
  }

  override def update(indexes: Array[Int], value: T): Unit = {
    require(indexes.length == this.nDimension, "invalid size")
    var offset = this._storageOffset
    var d = 0
    while (d < indexes.length) {
      offset += getOffset(indexes(d), d)
      d += 1
    }
    this._storage(offset) = value
  }

  override def isContiguous: Boolean = {
    DenseTensor.isContiguous(this)
  }

  override def contiguous(): Tensor[T] = {
    DenseTensor.newContiguous(this)
  }

  override def clone(): Tensor[T] = {
    DenseTensor.newClone(this)
  }

  override def shallowClone(): Tensor[T] = {
    Tensor(Storage(this.storage().array()), storageOffset(), size(), stride())
  }

  override def isSameSizeAs(other: Tensor[_]): Boolean = {
    DenseTensor.isSameSizeAs(this, other)
  }

  override def resize(sizes: Array[Int], strides: Array[Int]): Tensor[T] = {
    DenseTensor.resize(this, sizes, strides)
    this
  }

  override def resizeAs(src: Tensor[_]): Tensor[T] = {
    DenseTensor.resizeAs(this, src)
    this
  }

  override def resize(size: Int): Tensor[T] = {
    if (this.nDimension != 1 || this.size(0) != size) {
      DenseTensor.resize(this, Array(size))
    } else {
      this
    }
  }

  override def resize(size1: Int, size2: Int): Tensor[T] = {
    if (this.nDimension != 2 || this.size(0) != size1 || this.size(1) != size2) {
      DenseTensor.resize(this, Array(size1, size2))
    } else {
      this
    }
  }

  override def resize(size1: Int, size2: Int, size3: Int): Tensor[T] = {
    if (this.nDimension != 3 || this.size(0) != size1 || this.size(1) != size2 ||
        this.size(2) != size3) {
      DenseTensor.resize(this, Array(size1, size2, size3))
    } else {
      this
    }
  }

  override def unfold(dim: Int, size: Int, step: Int): Tensor[T] = {
    require(this.nDimension > 0, "cannot unfold an empty tensor")
    require(dim >= 0 && dim < this.nDimension, "out of arange")
    require(size <= this.size(dim), "out of arange")
    require(step > 0, "invalid step")

    val newSize = new Array[Int](this.nDimension + 1)
    val newStride = new Array[Int](this.nDimension + 1)

    newSize(this.nDimension) = size
    newStride(this.nDimension) = this.stride(dim)

    var d = 0
    while (d < this.nDimension) {
      if (d == dim) {
        newSize(d) = (this.size(d) - size) / step + 1
        newStride(d) = step * this.stride(d)
      } else {
        newSize(d) = this.size(d)
        newStride(d) = this.stride(d)
      }
      d = d + 1
    }
    new DenseTensor(this._storage, this._storageOffset, newSize, newStride, this.dim() + 1)
  }

  override def repeatTensor(sizes: Array[Int]): Tensor[T] = {
    require(sizes.length >= this.nDimension,
      "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor")
    val result = new DenseTensor[T]()
    val xTensor = this.clone()
    var xSize = xTensor.size()
    var i = 0
    while (i < sizes.length - this.dim()) {
      xSize = Array(1) ++ xSize
      i += 1
    }
    val size = new DenseTensor(new ArrayStorage[T](xSize.map(x => ev.fromType[Int](x))))
      .cmul(new DenseTensor(new ArrayStorage[T](sizes.map(x => ev.fromType[Int](x)))))
      .storage().array().map(x => ev.toType[Int](x))
    xTensor.resize(xSize)
    result.resize(size)
    var urTensor = Tensor(result)

    i = 0
    while (i < xTensor.dim()) {
      urTensor = urTensor.unfold(i, xTensor.size(i), xTensor.size(i))
      i += 1
    }

    i = 0
    while (i < urTensor.dim() - xTensor.dim()) {
      xSize = Array(1) ++ xSize
      i += 1
    }

    xTensor.resize(xSize)
    val xxTensor = xTensor.expandAs(urTensor)
    // println("xxTensor: ", xxTensor.size().mkString(" "), "  ", xxTensor.stride().mkString(" "),
    //  "  ", xxTensor.storage().array().mkString(" "))
    // println("urTensor: ", urTensor.size().mkString(" "), "  ", urTensor.stride().mkString(" "),
    //  "  ", urTensor.storage().array().mkString(" "))
    // println("result: ", result.size().mkString(" "), "  ", result.stride().mkString(" "),
    //  "  ", result.storage().array().mkString(" "))

    urTensor.copy(xxTensor)
    result
  }

  override def expandAs(template: Tensor[T]): Tensor[T] = {
    this.expand(template.size())
  }

  override def expand(sizes: Array[Int]): Tensor[T] = {
    require(sizes.length == this.dim(),
      s"the number of dimensions provided must equal ${this.dim()}")
    val tensorDim = this.dim()
    val tensorStride = this.stride()
    val tensorSize = this.size()

    var i = 0
    while (i < tensorDim) {
      if (tensorSize(i) == 1) {
        tensorSize(i) = sizes(i)
        tensorStride(i) = 0
      } else if (tensorSize(i) != sizes(i)) {
        throw new UnsupportedOperationException(
          "incorrect size: only supporting singleton expansion (size=1)")
      }
      i += 1
    }

    set(this.storage(), this.storageOffset(), tensorSize, tensorStride)
  }

  override def squeeze(): Tensor[T] = DenseTensor.squeeze(this)

  override def squeeze(dim: Int): Tensor[T] = DenseTensor.squeeze(this, dim)

  override def view(sizes: Array[Int]): Tensor[T] = {
    require(this.isContiguous, "current tensor is not contiguous")
    require(sizes.product == this.nElement(), "invalid size Element")
    new DenseTensor(this._storage, this.storageOffset(), sizes.clone())
  }

  override def toMLlibVector: Vector = {
    require(this.nDimension == 1, "tensor is not 1D")
    require(this.stride(0) == 1, "tensor is not continuous")
    new DenseVector(this.storage().array().asInstanceOf[Array[Double]])
  }

  override def toMLlibMatrix: Matrix = {
    require(this.nDimension == 2, "tensor is not 2D")
    require((this.stride(0) == 1 && this.stride(1) == this.size(0))
      || (this.stride(0) == this.size(1) && this.stride(1) == 1), "tensor is not continuous")
    new DenseMatrix(this.size(0), this.size(1), this.storage().array().asInstanceOf[Array[Double]],
      this.stride(1) == 1) // column major
  }

  override def getType: TensorDataType = ev.getType

  override def getTensorType: TensorType = DenseType

  override def reshape(sizes: Array[Int]): Tensor[T] = {
    require(sizes.product == this.nElement(),
      "DenseTensor: nElement of this tensor is not equal to nElement specified by sizes")
    val result = new DenseTensor[T]()
    result.resize(sizes)
    result.copy(this)
    result
  }

  override def save(path: String, overWrite: Boolean): this.type = {
    File.save(this, path, overWrite)
    this
  }

  override def set(other: Tensor[T]): Tensor[T] = {
    require(other.isInstanceOf[DenseTensor[_]], "Only support dense tensor in this operation")
    DenseTensor.rawSet(this, other.storage().asInstanceOf[ArrayStorage[T]],
      other.storageOffset(), other.nDimension, other.size(), other.stride())
  }

  override def set(storage: Storage[T], storageOffset: Int = 0, sizes: Array[Int] = null,
      strides: Array[Int] = null): Tensor[T] = {
    if (sizes != null && strides != null) {
      require(sizes.length == strides.length)
    }
    require(storage.isInstanceOf[ArrayStorage[_]], "Only support array storage in this operation")
    //noinspection ScalaUnnecessaryParentheses
    DenseTensor.rawSet(this, storage.asInstanceOf[ArrayStorage[T]], storageOffset,
      (if (sizes == null) 0 else sizes.length), sizes, strides)
  }

  override def set(): Tensor[T] = {
    if (this._storage != null) {
      this._storage.resize(0)
    }
    this.nDimension = 0
    this._size = Array.emptyIntArray
    this
  }

  override def addmv(alpha: T, mat: Tensor[T], vec: Tensor[T]): Tensor[T] = {
    DenseTensorMath.addmv(this, ev.fromType[Int](1), this, alpha, mat, vec)
  }

  override def addmv(beta: T, alpha: T, mat: Tensor[T], vec: Tensor[T]): Tensor[T] = {
    DenseTensorMath.addmv(this, beta, this, alpha, mat, vec)
  }

  override def addmm(v1: T, M: Tensor[T], v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    DenseTensorMath.addmm(this, v1, M, v2, mat1, mat2)
  }

  override def addmm(v1: T, v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    DenseTensorMath.addmm(this, v1, this, v2, mat1, mat2)
  }

  override def addmm(v: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    DenseTensorMath.addmm[T](this, ev.fromType[Int](1), this, v, mat1, mat2)
  }

  override def bmm(beta: T, alpha: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T] = {
    DenseTensorMath.bmm(this, beta, this, alpha, mat1, mat2)
  }

  override def addr(v: T, vec1: Tensor[T], vec2: Tensor[T]): Tensor[T] = {
    DenseTensorMath.addr(this, ev.fromType[Int](1), this, v, vec1, vec2)
  }

  override def mul(x: Tensor[T], value: T): Tensor[T] = DenseTensorMath.mul(this, x, value)

  override def mul(value: T): Tensor[T] = DenseTensorMath.mul(this, null, value)

  override def div(value: T): Tensor[T] = DenseTensorMath.mul(this, null, ev.inv(value))

  override def cmul(y: Tensor[T]): Tensor[T] = {
    require(y.isInstanceOf[DenseTensor[_]], "Only support dense tensor in this operation")
    DenseTensorMath.cmul(this, this, y.asInstanceOf[DenseTensor[T]])
  }

  override def cmul(x: Tensor[T], y: Tensor[T]): Tensor[T] = {
    DenseTensorMath.cmul(this, x.asInstanceOf[DenseTensor[T]], y.asInstanceOf[DenseTensor[T]])
  }

  override def cdiv(y: Tensor[T]): Tensor[T] = DenseTensorMath.cdiv(this, this, y)

  override def cdiv(x: Tensor[T], y: Tensor[T]): Tensor[T] = DenseTensorMath.cdiv(this, x, y)

  override def addcdiv(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T] = {
    ev.addcdiv(value, this.nElement(), this.storage().array(), this.storageOffset(),
      tensor1.storage().array(), tensor1.storageOffset(),
      tensor2.storage().array(), tensor2.storageOffset())
    this
  }

  override def add(value: T, y: Tensor[T]): Tensor[T] = DenseTensorMath.cadd(this, this, value, y)

  override def add(value: T): Tensor[T] = {
    ev.add(this.nElement(), this.storage().array(), this.storageOffset(), value, 1)
    this
  }

  override def add(y: Tensor[T]): Tensor[T] = {
    ev.vAdd(this.nElement(), this.storage().array(), this.storageOffset(),
      y.storage().array(), y.storageOffset(), this.storage().array(),
      this.storageOffset())
    this
  }

  override def sub(value: T): Tensor[T] = {
    ev.sub(this.nElement(), this.storage().array(), this.storageOffset(), value, 1)
    this
  }

  override def sub(y: Tensor[T]): Tensor[T] = {
    ev.vSub(this.nElement(), this.storage().array(), this.storageOffset(),
      y.storage().array(), y.storageOffset(), this.storage().array(),
      this.storageOffset())
    this
  }

  override def log(): Tensor[T] = DenseTensorMath.log[T](this, this)

  override def pow(x: Tensor[T], n: T): Tensor[T] = DenseTensorMath.pow[T](this, x, n)

  override def pow(n: T): Tensor[T] = DenseTensorMath.pow[T](this, this, n)

  override def exp(): Tensor[T] = DenseTensorMath.exp[T](this, this)

  override def exp(x: Tensor[T]): Tensor[T] = DenseTensorMath.exp[T](this, x)

  override def sqrt(x: Tensor[T]): Tensor[T] = DenseTensorMath.sqrt[T](this, x)

  override def dot(y: Tensor[T]): T = {
    require(this.nElement() == y.nElement())
    ev.dot(this.nElement(), this.storage().array(), this.storageOffset(), 1,
      y.storage().array(), y.storageOffset(), 1)
  }

  override def sum(): T = DenseTensorMath.sumAll(this)

  override def sum(dim: Int): Tensor[T] = DenseTensorMath.sum(null, this, dim)

  override def max(): T = DenseTensorMath.maxAll(this)
  /*
  override def range(xmin: Double, xmax: Double, step: Int = 1): Tensor[T] = {
    require(xmax >= xmin && step > 0)
    val size = math.floor((xmax - xmin) / step).toInt
    if (this.nElement() != size) this.resize(size)
    var i = 0
    val func = new TensorFunc2[T] {
      override def apply(data1: Array[T], offset1: Int): Unit = {
        data1(offset1) = ev.fromType(xmin + i * step)
        i += 1
      }
    }
    DenseTensorApply.apply1[T](this, func)
    this
  }
  */

  override def select(dim: Int, index: Int): DenseTensor[T] = {
    val _dimension = dim
    val _sliceIndex = index

    require(this.nDimension >= 0, "empty or scalar tensor cannot be selected")
    val result = DenseTensor.newWithTensor(this)
    DenseTensor.select(result, null, _dimension, _sliceIndex)
    result
  }

  override def narrow(dim: Int, index: Int, size: Int): DenseTensor[T] = {
    /*
    val _dim = dim
    val _index = index
    require(_dim >= 0 && _dim < nDimension, "dimension out of range")
    require(_index >= 0 && _index < this._size(_dim), "index out of range")
    require(size > 0 && _index + size <= this._size(_dim), "size out of range")

    val newTensor = DenseTensor.newWithTensor(this)
    if (_index > 0) {
      newTensor._storageOffset += (_index * _stride(_dim))
    }
    newTensor._size(_dim) = size
    newTensor
    */

    val result = DenseTensor.newWithTensor(this)
    DenseTensor.narrow(result, null, dim, index, size)
    result
  }

  override def copy(other: Tensor[T]): Tensor[T] = {
    DenseTensor.copy(this, other)
    this
  }

  private def getOffset(z: Int, dim: Int): Int = {
    var _z = z
    if (_z < 0) {
      _z = this.size(dim) + _z
    }
    require(_z >= 0 && _z < this.size(dim), "index out of bound")
    _z * this.stride(dim)
  }

  override def toTensor[D](implicit ev2: TensorNumeric[D]): Tensor[D] = {
    if (ev.getType == ev2.getType) {
      this.asInstanceOf[Tensor[D]]
    } else {
      throw new IllegalArgumentException
    }
  }
}

object DenseTensor {

  def apply[@specialized(Float, Double) T: ClassTag](value: T)(
      implicit ev: TensorNumeric[T]): Tensor[T] = {
    new DenseTensor[T](new ArrayStorage[T](Array(value)), 0, Array[Int](), Array[Int](), 0)
  }

  private[tensor] def squeeze[@specialized(Float, Double) T](self: DenseTensor[T]): Tensor[T] = {
    var ndim = 0
    var d = 0
    while (d < self.nDimension) {
      if (self._size(d) != 1) {
        if (d != ndim) {
          self._size(ndim) = self._size(d)
          self._stride(ndim) = self._stride(d)
        }
        ndim += 1
      }
      d += 1
    }

    if (ndim == 0 && self.nDimension > 0) {
      self._size(0) = 1
      self._stride(0) = 1
      ndim = 1
    }
    self.nDimension = ndim
    // self._size = self._size.slice(0, ndim)
    self
  }

  private[tensor] def squeeze[@specialized(Float, Double) T](self: DenseTensor[T], _dim: Int): Tensor[T] = {
    require(_dim >= 0 && _dim < self.nDimension, "dimension out of arange")
    if (self._size(_dim) == 1 && self.nDimension > 1) {
      var d = _dim
      while (d < self.nDimension - 1) {
        self._size(d) = self._size(d + 1)
        self._stride(d) = self._stride(d + 1)
        d += 1
      }
      self.nDimension -= 1
    }
    self
  }

  private[tensor] def select[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T],
      source: DenseTensor[T], _dimension: Int, _sliceIndex: Int): Unit = {
    var src = source
    if (src == null) src = self
    require(src.nDimension > 0, "cannot select on a scalar")
    require(_dimension >= 0 && _dimension < src.nDimension, "out of range")
    require(_sliceIndex >= 0 && _sliceIndex < src.size(_dimension),
      s"${_sliceIndex} out of range 0 to ${src.size(_dimension) - 1}")

    set(self, src)
    narrow(self, null, _dimension, _sliceIndex, 1)

    var d = _dimension
    while (d < self.nDimension - 1) {
      self._size(d) = self._size(d + 1)
      self._stride(d) = self._stride(d + 1)
      d += 1
    }

    self.nDimension = self.nDimension - 1
  }

  private[tensor] def narrow[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T],
      source: DenseTensor[T], _dimension: Int, _firstIndex: Int, size: Int): Unit = {
    var src = source
    if (src == null) {
      src = self
    }

    require(_dimension >= 0 && _dimension < src.nDimension, "dimension out of range")
    require(_firstIndex >= 0 && _firstIndex < src.size(_dimension),
      s"firstIndex(${_firstIndex}) out of range [0, ${src.size(_dimension)})")
    require(size > 0 && _firstIndex + size <= src.size(_dimension),
      s"size out of range $size (0, ${src.size(_dimension)} - ${_firstIndex}]")

    set(self, src)

    if (_firstIndex > 0) {
      self._storageOffset = self._storageOffset + _firstIndex * self._stride(_dimension)
    }
    self._size(_dimension) = size
  }

  private[tensor] def set[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T],
      other: DenseTensor[T]): Tensor[T] = {
    if (self != other) {
      DenseTensor.rawSet(self, other.storage().asInstanceOf[ArrayStorage[T]], other.storageOffset(),
        other.nDimension, other.size(), other.stride())
    } else {
      self
    }
  }

  private[tensor] def newWithTensor[@specialized(Float, Double) T: ClassTag](other: DenseTensor[T])(
      implicit ev: TensorNumeric[T]): DenseTensor[T] = {
    val self = new DenseTensor[T]()
    DenseTensor.rawSet[T](self, other._storage, other._storageOffset,
      other.nDimension, other._size, other._stride)
  }

  private[tensor] def isContiguous[@specialized(Float, Double) T](self: DenseTensor[T]): Boolean = {
    var s = 1
    var d = self.nDimension - 1
    var res = true
    while (d >= 0 && res) {
      if (self._size(d) != 1) {
        if (s != self._stride(d)) {
          res = false
        } else {
          s *= self._size(d)
        }
      }
      d -= 1
    }
    res
  }

  private[tensor] def newClone[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T])(
      implicit ev: TensorNumeric[T]): DenseTensor[T] = {
    val newTensor = new DenseTensor[T]()
    resizeAs(newTensor, self)
    copy(newTensor, self)
    newTensor
  }

  private[tensor] def newContiguous[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T])(
      implicit ev: TensorNumeric[T]): DenseTensor[T] = {
    if (!isContiguous(self)) {
      newClone(self)
    } else {
      self
    }
  }

  private[tensor] def isSameSizeAs[@specialized T](self: DenseTensor[T], src: Tensor[_]): Boolean = {
    var res = if (self.nDimension != src.nDimension || self.isEmpty != src.isEmpty) false else true
    var d = 0
    while (d < self.nDimension && res) {
      if (self.size(d) != src.size(d)) {
        res = false
      }
      d += 1
    }
    res
  }

  private[tensor] def rawSet[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T],
      storage: ArrayStorage[T], storageOffset: Int, nDimension: Int, _size: Array[Int],
      _stride: Array[Int]): DenseTensor[T] = {
    self._storage = storage
    require(storageOffset >= 0, "storageOffset must >= 0")
    self._storageOffset = storageOffset
    rawResize[T](self, nDimension, _size, _stride)
  }

  private[tensor] def rawResize[@specialized(Float, Double) T: ClassTag](self: DenseTensor[T],
      nDim: Int, _size: Array[Int], _stride: Array[Int]): DenseTensor[T] = {

    // scalar condition
    if (nDim == 0 && _size.isEmpty) {
      self._size = Array.emptyIntArray
      self._stride = Array.emptyIntArray
      self.nDimension = 0
      val newSize = self._storageOffset + 1
      if (self._storage == null) {
        self._storage = new ArrayStorage[T](new Array[T](newSize))
      } else if (newSize > self._storage.length) {
        self._storage.resize(newSize)
      }
      return self
    }

    val isSameState = {
      if (nDim != self.nDimension) {
        false
      } else {
        var tmp = true
        var d = 0
        while (d < nDim) {
          if (_size(d) != self._size(d))
            tmp = false
          if (_stride != null && _stride(d) >= 0 && _stride(d) != self._stride(d))
            tmp = false
          d += 1
        }
        tmp
      }
    }
    if (isSameState) return self

    if (self.nDimension != nDim) {
      self._size = new Array[Int](nDim)
      self._stride = new Array[Int](nDim)
      self.nDimension = nDim
    }
    var totalSize = 1
    var d = self.nDimension - 1
    while (d >= 0) {
      self._size(d) = _size(d)
      if (_stride != null && _stride(d) >= 0) {
        self._stride(d) = _stride(d)
      } else {
        self._stride(d) = {
          if (d == self.nDimension - 1) 1 else self._size(d + 1) * self._stride(d + 1)
        }
      }
      totalSize += (self._size(d) - 1) * self._stride(d)
      d -= 1
    }

    val newSize = totalSize + self._storageOffset
    if (newSize > 0) {
      if (self._storage == null) {
        self._storage = new ArrayStorage(new Array[T](newSize))
      } else if (newSize > self._storage.length) {
        self._storage.resize(newSize)
      }
    }
    self
  }

  private[tensor] def resizeAs[@specialized(Float, Double) T: ClassTag](
      self: DenseTensor[T], src: Tensor[_]): Unit = {
    if (!isSameSizeAs(self, src)) {
      rawResize(self, src.nDimension, src.size(), null)
    }
  }

  private[tensor] def resize[@specialized(Float, Double) T: ClassTag](
      self: DenseTensor[T], sizes: Array[Int], strides: Array[Int] = null): DenseTensor[T] = {
    require(sizes != null, "invalid size")
    if (strides != null) {
      require(sizes.length == strides.length, "invalid stride")
    }
    rawResize(self, sizes.length, sizes, strides)
  }

  private[tensor] def newWithStorage[@specialized(Float, Double) T: ClassTag](
      tensor: DenseTensor[T], storage: ArrayStorage[T], storageOffset: Int, size: Array[Int],
      stride: Array[Int], ev: TensorNumeric[T]): DenseTensor[T] = {
    if (size != null && stride != null) {
      require(size.length == stride.length, "inconsistent size")
    }

    implicit val ev2: TensorNumeric[T] = ev
    val self = if (tensor == null) new DenseTensor[T]() else tensor
    val nDimension = if (size != null) size.length else if (stride != null) stride.length else 0

    DenseTensor.rawSet[T](self, storage, storageOffset, nDimension, size, stride)
  }

  /*
  private[tensor] def range[@specialized(Float, Double) T: ClassTag](xmin: Double, xmax: Double,
      step: Int = 1)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val newTensor = Tensor[T]()
    newTensor.range(xmin, xmax, step)
  }
  */

  private[tensor] def copy[@specialized T](self: DenseTensor[T], src: Tensor[T]): Unit = {
    require(self.nElement() == src.nElement(), "tensor total elements don't match")
    if (self.isContiguous && src.isContiguous && sameStride(self.stride(), src.stride())) {
      System.arraycopy(
        src.storage().array(),
        src.storageOffset(),
        self.storage().array(),
        self.storageOffset(),
        self.nElement())
    } else {
      val func2 = new TensorFunc4[T] {
        override def apply(data1: Array[T], offset1: Int, data2: Array[T], offset2: Int): Unit = {
          data1(offset1) = data2(offset2)
        }
      }
      DenseTensorApply.apply2[T](self, src, func2)
    }
  }

  private[tensor] def sameStride(l: Array[Int], r: Array[Int]): Boolean = {
    if (l.length != r.length) return false
    var i = 0
    while (i < l.length) {
      if (l(i) != r(i)) {
        return false
      }
      i += 1
    }
    return true
  }

  private[tensor] def size2Stride(sizes: Array[Int]): Array[Int] = {
    val strides = new Array[Int](sizes.length)
    var jump = 1
    var i = strides.length - 1
    while (i >= 0) {
      strides(i) = jump
      jump = jump * sizes(i)
      i -= 1
    }
    strides
  }

  private[tensor] def canFastBroadcast[T](tensor: Tensor[T], other: Tensor[T]): Boolean = {
    if (tensor.nDimension < other.nDimension) return false
    val delta = tensor.nDimension - other.nDimension
    var d = other.nDimension - 1
    var broadcasting = false
    while (d >= 0) {
      if (broadcasting) {
        if (other.size(d) != 1) return false
      } else if (tensor.size(delta + d) != other.size(d)) {
        if (other.size(d) != 1) return false
        broadcasting = true
      }
      d -= 1
    }
    true
  }
}
