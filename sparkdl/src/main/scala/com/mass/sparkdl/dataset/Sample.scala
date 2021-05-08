package com.mass.sparkdl.dataset

import scala.reflect.ClassTag

import org.apache.commons.lang3.SerializationUtils
import com.mass.sparkdl.tensor.{Storage, Tensor, TensorNumeric}

abstract class Sample[T: ClassTag] extends Serializable {

  def featureLength(): Int

  def labelLength(): Int

  // def numFeature(): Int

  // def numLabel(): Int

  override def clone(): this.type = SerializationUtils.clone(this)

  def feature()(implicit ev: TensorNumeric[T]): Tensor[T]

  // def feature(index: Int)(implicit ev: TensorNumeric[T]): Tensor[T]

  def label()(implicit ev: TensorNumeric[T]): Tensor[T]

  // def label(index: Int)(implicit ev: TensorNumeric[T]): Tensor[T]

  def getFeatureSize: Array[Int] // Array[Array[Int]]

  def getLabelSize: Array[Int] // Array[Array[Int]]

  def getData: Array[T]
}

private[sparkdl] class ArraySample[T: ClassTag](
    private val data: Array[T],
    private val featureSize: Array[Int],
    private val labelSize: Array[Int]) extends Sample[T] {

  require(data != null, "data couldn't be empty")
  require(featureSize != null, "feature couldn't be empty")

  override def getData: Array[T] = data

  override def featureLength(): Int = {
    featureSize(0)
  }

  override def labelLength(): Int = {
    if (null != labelSize && labelSize.nonEmpty) {
      labelSize(0)
    } else {
      0
    }
  }

  override def getFeatureSize: Array[Int] = featureSize

  override def getLabelSize: Array[Int] = {
    require(null != labelSize, "Sample doesn't have label")
    labelSize
  }

  override def feature()(implicit ev: TensorNumeric[T]): Tensor[T] = {
    Tensor[T](Storage(data), 0, featureSize)
  }

  override def label()(implicit ev: TensorNumeric[T]): Tensor[T] = {
    if (null != labelSize && labelSize.nonEmpty) {
      val labelOffset = featureSize.product
      Tensor[T](Storage(data), labelOffset, labelSize)
    } else {
      null
    }
  }

  def canEqual(other: Any): Boolean = other.isInstanceOf[ArraySample[T]]

  override def equals(other: Any): Boolean = other match {
    case that: ArraySample[T] =>
      if (!that.canEqual(this) || data.deep != that.data.deep ||
        featureSize.deep != that.featureSize.deep)
        return false
      if (null != labelSize && null != that.labelSize) {
        labelSize.deep == that.labelSize.deep
      } else {
        null == labelSize && null == that.labelSize
      }
    case _ => false
  }

  override def hashCode(): Int = {
    val state = if (null == labelSize) {
      Seq(data, featureSize)
    } else {
      Seq(data, featureSize, labelSize)
    }
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object ArraySample {

  def apply[T: ClassTag](
      data: Array[T],
      featureSize: Array[Int],
      labelSize: Array[Int]): Sample[T] = {
    new ArraySample(data, featureSize, labelSize)
  }

  def apply[T: ClassTag](featureTensor: Tensor[T], labelTensor: Tensor[T])(
      implicit ev: TensorNumeric[T]): Sample[T] = {
    val data = new Array[T](featureTensor.nElement() + labelTensor.nElement())
    ev.arraycopy(featureTensor.storage().array(), featureTensor.storageOffset(),
      data, 0, featureTensor.nElement())
    ev.arraycopy(labelTensor.storage().array(), labelTensor.storageOffset(),
      data, featureTensor.nElement(), labelTensor.nElement())
    new ArraySample[T](data, getSize(featureTensor), getSize(labelTensor))
  }

  def apply[T: ClassTag](featureTensor: Tensor[T], label: T)(
      implicit ev: TensorNumeric[T]) : Sample[T] = {
    val data = new Array[T](featureTensor.nElement() + 1)
    ev.arraycopy(featureTensor.storage().array(), featureTensor.storageOffset(),
      data, 0, featureTensor.nElement())
    data(featureTensor.nElement()) = label
    new ArraySample[T](data, getSize(featureTensor), Array(1))
  }

  def apply[T: ClassTag](featureTensor: Tensor[T])(
      implicit ev: TensorNumeric[T]) : Sample[T] = {
    val data = new Array[T](featureTensor.nElement())
    ev.arraycopy(featureTensor.storage().array(), featureTensor.storageOffset(),
      data, 0, featureTensor.nElement())
    new ArraySample[T](data, getSize(featureTensor), null)
  }

  private def copy[T: ClassTag](data: Array[T], tensors: Array[Tensor[T]])(
      implicit ev: TensorNumeric[T]): Array[T] = {
    var offset = 0
    var i = 0
    while (i < tensors.length) {
      val tensor = tensors(i)
      require(tensor.isContiguous, s"$i-th tensor is not contiguous")
      ev.arraycopy(tensor.storage().array(), tensor.storageOffset(),
        data, offset, tensor.nElement())
      offset += tensor.nElement()
      i += 1
    }
    data
  }

  private def getSize[T: ClassTag](tensors: Array[Tensor[T]]): Array[Array[Int]] = {
    tensors.map(_.size())
  }

  private def getSize[T: ClassTag](tensor: Tensor[T]): Array[Int] = {
    tensor.size()
  }

  private def sameSize(a: Array[Array[Int]], b: Array[Array[Int]]): Boolean = {
    if (a.length != b.length) return false
    var i = 0
    while (i < a.length) {
      if (a(i).length != b(i).length) return false
      i += 1
    }
    true
  }
}

object Sample {

  def apply[T: ClassTag](
      data: Array[T],
      featureSize: Array[Int],
      labelSize: Array[Int]): Sample[T] = {
    ArraySample(data, featureSize, labelSize)
  }

  def apply[T: ClassTag](
      featureTensor: Tensor[T],
      labelTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    ArraySample(featureTensor, labelTensor)
  }

  def apply[T: ClassTag](
      featureTensor: Tensor[T],
      label: T)(implicit ev: TensorNumeric[T]) : Sample[T] = {
    ArraySample(featureTensor, label)
  }

  def apply[T: ClassTag](
      featureTensor: Tensor[T])(implicit ev: TensorNumeric[T]) : Sample[T] = {
    ArraySample(featureTensor)
  }
}

private[sparkdl] class TensorSample[T: ClassTag](
    val features: Tensor[T],
    val labels: Tensor[T]) extends Sample[T] {

  protected val featureSize: Array[Int] = features.size()
  protected val labelSize: Array[Int] = labels.size()

  override def featureLength(): Int = {
    features.size(0)
  }

  override def labelLength(): Int = {
    labels.size(0)
  }

  override def getFeatureSize: Array[Int] = {
    featureSize
  }

  override def getLabelSize: Array[Int] = {
    labelSize
  }

  override def getData: Array[T] = {
    throw new UnsupportedOperationException()
  }

  override def feature()(implicit ev: TensorNumeric[T]): Tensor[T] = {
    this.feature
  }

  override def label()(implicit ev: TensorNumeric[T]): Tensor[T] = {
    if (null != this.labels) this.label else null
  }
}
