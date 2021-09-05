package com.mass.sparkdl.tensor

import java.io.Serializable

import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.Activity
import com.mass.sparkdl.utils.Table
import org.apache.commons.math3.random.RandomDataGenerator
import org.apache.spark.mllib.linalg.{Matrix, Vector}

trait Tensor[T] extends Serializable with TensorMath[T] with Activity {

  def isEmpty: Boolean

  def isScalar: Boolean

  def nDimension: Int

  def dim(): Int

  def size(): Array[Int]

  def size(dim: Int): Int

  def stride(): Array[Int]

  def stride(dim: Int): Int

  def fill(v: T): Tensor[T]

  def zero(): Tensor[T]

  // def randn(seed: Long): Tensor[T]

  def randn(mean: Double, stdv: Double, seed: Long = System.nanoTime()): Tensor[T]

  def rand(lowerBound: Double, upperBound: Double, seed: Long = System.nanoTime()): Tensor[T]

  def randInt(lowerBound: Int, upperBound: Int, seed: Long = System.nanoTime()): Tensor[T]

  def transpose(dim1: Int, dim2: Int): Tensor[T]

  def t(): Tensor[T]

  def apply(index: Int): Tensor[T]

  def apply(indexes: Array[Int]): T

  def value(): T

  def valueAt(d1: Int): T

  def valueAt(d1: Int, d2: Int): T

  def setValue(value: T): this.type

  def setValue(d1: Int, value: T): this.type

  def setValue(d1: Int, d2: Int, value: T): this.type

  def update(index: Int, value: T): Unit

  // def update(index: Int, src: Tensor[T]): Unit

  def update(indexes: Array[Int], value: T): Unit

  // def update(filter: T => Boolean, value: T): Unit

  def isContiguous: Boolean

  def contiguous(): Tensor[T]

  def isSameSizeAs(other: Tensor[_]): Boolean

  override def clone(): Tensor[T] = this

  def shallowClone(): Tensor[T]

  def resizeAs(src: Tensor[_]): Tensor[T]

  // def cast[D: ClassTag](castTensor: Tensor[D]): Tensor[D]

  def resize(sizes: Array[Int], strides: Array[Int] = null): Tensor[T]

  def resize(size: Int): Tensor[T]

  def resize(size1: Int, size2: Int): Tensor[T]

  def resize(size1: Int, size2: Int, size3: Int): Tensor[T]

  def unfold(dim: Int, size: Int, step: Int): Tensor[T]

  def repeatTensor(sizes: Array[Int]): Tensor[T]

  def expandAs(template: Tensor[T]): Tensor[T]

  def expand(sizes: Array[Int]): Tensor[T]

  def nElement(): Int

  def select(dim: Int, index: Int): Tensor[T]

  def storage(): Storage[T]

  def storageOffset(): Int

  def narrow(dim: Int, index: Int, size: Int): Tensor[T]

  def copy(other: Tensor[T]): Tensor[T]

  def squeeze(): Tensor[T]

  def squeeze(dim: Int): Tensor[T]

  def view(sizes: Int*): Tensor[T] = view(sizes.toArray)

  def view(sizes: Array[Int]): Tensor[T]

  def set(other: Tensor[T]): Tensor[T]

  def set(storage: Storage[T], storageOffset: Int = 0, sizes: Array[Int] = null,
          strides: Array[Int] = null): Tensor[T]

  def set(): Tensor[T]

  def toMLlibVector: Vector

  def toMLlibMatrix: Matrix

  def getType: TensorDataType

  def reshape(sizes: Array[Int]): Tensor[T]

  def save(path: String, overWrite: Boolean = false): this.type

  def getTensorType: TensorType

  override def toTable: Table =
    throw new IllegalArgumentException("Tensor cannot be cast to Table")

  override def isTensor: Boolean = true

  override def isTable: Boolean = false
}

sealed trait TensorDataType extends Serializable

object BooleanType extends TensorDataType

object CharType extends TensorDataType

object ByteType extends TensorDataType

object StringType extends TensorDataType

object IntType extends TensorDataType

object ShortType extends TensorDataType

object LongType extends TensorDataType

object FloatType extends TensorDataType

object DoubleType extends TensorDataType

sealed trait TensorType

object DenseType extends TensorType

object SparseType extends TensorType

object MklDnnType extends TensorType

object Tensor {

  def apply[@specialized(Float, Double) T: ClassTag]()(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T]()

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T](d1)

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T](d1, d2)

  def apply[@specialized(Float, Double) T: ClassTag](d1: Int, d2: Int, d3: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor[T](d1, d2, d3)

  def apply[@specialized(Float, Double) T: ClassTag](dims: Int*)(
      implicit ev: TensorNumeric[T]): Tensor[T] = {
    new DenseTensor[T](new ArrayStorage[T](new Array[T](dims.product)), 0,
      dims.toArray, DenseTensor.size2Stride(dims.toArray), dims.length)
  }

  def apply[@specialized(Float, Double) T: ClassTag](sizes: Array[Int])(
      implicit ev: TensorNumeric[T]): Tensor[T] =
    new DenseTensor(new ArrayStorage[T](new Array[T](sizes.product)), 0,
      sizes.clone(), DenseTensor.size2Stride(sizes.clone()), sizes.length)

  def apply[@specialized(Float, Double) T: ClassTag](storage: Storage[T])(
      implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(storage.isInstanceOf[ArrayStorage[_]], "Only support array storage in this operaiton")
    new DenseTensor(storage.asInstanceOf[ArrayStorage[T]])
  }

  def apply[@specialized(Float, Double) T: ClassTag](data: Array[T], shape: Array[Int])(
      implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(shape.product == data.length, "shape total size doesn't match data length")
    new DenseTensor[T]().set(Storage[T](data), storageOffset = 0, sizes = shape)
  }

  def apply[@specialized(Float, Double) T: ClassTag](
      storage: Storage[T],
      storageOffset: Int,
      size: Array[Int] = null,
      stride: Array[Int] = null)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    new DenseTensor(storage.asInstanceOf[ArrayStorage[T]], storageOffset, size, stride)
  }

  def apply[@specialized(Float, Double) T: ClassTag](other: Tensor[T])(
    implicit ev: TensorNumeric[T]): Tensor[T] = new DenseTensor(other)

  def arange[@specialized(Float, Double) T: ClassTag](xmin: Double, xmax: Double, step: Int = 1)(
      implicit ev: TensorNumeric[T]): Tensor[T] = {
    val size = math.floor((xmax - xmin) / step).toInt.abs
    Tensor((xmin until xmax by step).map(ev.fromType(_)).toArray, Array(size))
    // DenseTensor.arange[T](xmin, xmax, step)
  }

  def randn[@specialized(Float, Double) T: ClassTag](
      sizes: Array[Int],
      mean: Double,
      stdev: Double,
      seed: Long = System.nanoTime())(
      implicit ev: TensorNumeric[T]): Tensor[T] = {

    val generator = new RandomDataGenerator()
    generator.reSeed(seed)
    var i = 0
    val res = Tensor[T](sizes)
    val total = res.nElement()
    val data = res.storage().array()
    val offset = res.storageOffset()
    while (i < total) {
      data(offset + i) = ev.fromType(generator.nextGaussian(mean, stdev))
      i += 1
    }
    res
  }

  def rand[@specialized(Float, Double) T: ClassTag](
    sizes: Array[Int],
    lowerBound: Double,
    upperBound: Double,
    seed: Long = System.nanoTime())(
    implicit ev: TensorNumeric[T]): Tensor[T] = {

    val generator = new RandomDataGenerator()
    generator.reSeed(seed)
    var i = 0
    val res = Tensor[T](sizes)
    val total = res.nElement()
    val data = res.storage().array()
    val offset = res.storageOffset()
    while (i < total) {
      data(offset + i) = ev.fromType(generator.nextUniform(lowerBound, upperBound))
      i += 1
    }
    res
  }

  def randInt(
      sizes: Array[Int],
      lowerBound: Int,
      upperBound: Int,
      seed: Long = System.nanoTime()): Tensor[Int] = {

    val generator = new RandomDataGenerator()
    generator.reSeed(seed)
    var i = 0
    val res = Tensor[Int](sizes)
    val total = res.nElement()
    val data = res.storage().array()
    val offset = res.storageOffset()
    while (i < total) {
      data(offset + i) = generator.nextInt(lowerBound, upperBound - 1)
      i += 1
    }
    res
  }

  def randLong(
    sizes: Array[Int],
    lowerBound: Int,
    upperBound: Int,
    seed: Long = System.nanoTime()): Tensor[Long] = {

    val generator = new RandomDataGenerator()
    generator.reSeed(seed)
    var i = 0
    val res = Tensor[Long](sizes)
    val total = res.nElement()
    val data = res.storage().array()
    val offset = res.storageOffset()
    while (i < total) {
      data(offset + i) = generator.nextInt(lowerBound, upperBound - 1)
      i += 1
    }
    res
  }

  def unique[T: ClassTag](tensor: Tensor[T])(implicit ev: TensorNumeric[T]): (Tensor[T], Tensor[Int]) = {
    require(tensor.isContiguous, "unqiue only supports contiguous tensor")
    require(tensor.dim() == 1, "unique only supports 1D tensor")
    val array = tensor.storage().array()
    val arrayOffset = tensor.storageOffset()
    val distinctTensor = Tensor().resizeAs(tensor)
    val tensorIndices = Tensor[Int]().resizeAs(tensor)

    val distinctValues = distinctTensor.storage().array()
    val distinctValuesOffset = distinctTensor.storageOffset()
    val indicesArray = tensorIndices.storage().array()
    val indicesOffset = tensorIndices.storageOffset()
    val seen = scala.collection.mutable.HashMap[T, Int]()
    var i = 0
    var nonZero = 0
    while (i < tensor.nElement()) {
      val x = array(i + arrayOffset)
      if (!seen.contains(x)) {
        distinctValues(nonZero + distinctValuesOffset) = x
        seen(x) = nonZero
        nonZero += 1
      }
      indicesArray(i + indicesOffset) = seen(x)
      i += 1
    }
    distinctTensor.resize(nonZero)
    (distinctTensor, tensorIndices)
  }
}
