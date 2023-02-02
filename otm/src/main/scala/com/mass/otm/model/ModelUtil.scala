package com.mass.otm.model

import scala.reflect.ClassTag

import com.mass.otm.DeepModel
import com.mass.scalann.tensor.{Storage, Tensor, TensorNumeric}

object ModelUtil {

  def cumSum[A](xs: Seq[A])(implicit num: Numeric[A]): Seq[A] = {
    xs.tail.scanLeft(xs.head)(num.plus)
  }

  def compactParameters[T: ClassTag](model: DeepModel[T]): DeepModel[T] = {
    val (weightParams, gradParams) = model.parameters()
    weightParams.zip(gradParams).foreach { case (w, g) => g.resizeAs(w) }
    val length = weightParams.map(_.nElement()).sum
    val offsets = weightParams.map(_.nElement()).init.scanLeft(0)(_ + _)
    val weightStorage = Storage[T](length)
    val gradStorage = Storage[T](length)
    offsets lazyZip weightParams lazyZip gradParams foreach { case (offset, weight, gradient) =>
      copyParameter(weight, weightStorage, offset)
      copyParameter(gradient, gradStorage, offset)
    }
    model
  }

  def copyParameter[T](
      parameter: Tensor[T],
      storage: Storage[T],
      offset: Int
  ): Unit = {
    System.arraycopy(
      parameter.storage().array(),
      parameter.storageOffset(),
      storage.array(),
      offset,
      parameter.nElement()
    )
    parameter.set(
      storage,
      offset,
      parameter.size(),
      parameter.stride()
    )
  }

  // assume the parameters are compact.
  def getParameters[@specialized(Float, Double) T: ClassTag](model: DeepModel[T])(implicit
      ev: TensorNumeric[T]
  ): (Tensor[T], Tensor[T]) = {
    val (weightParams, gradParams) = model.parameters()
    val weightTensor = Tensor(
      weightParams.head.storage(),
      weightParams.head.storageOffset(),
      Array(weightParams.map(_.nElement()).sum)
    )
    val gradTensor = Tensor(
      gradParams.head.storage(),
      gradParams.head.storageOffset(),
      Array(gradParams.map(_.nElement()).sum)
    )
    (weightTensor, gradTensor)
  }

  // assume the parameters are compact.
  def extractWeights[T: ClassTag](model: DeepModel[T])(implicit
      ev: TensorNumeric[T]
  ): Seq[Tensor[T]] = {
    val (weightParams, _) = model.parameters()
    val storage = Storage[T](weightParams.head.storage().array())
    weightParams.toSeq.map { w =>
      Tensor[T](storage, w.storageOffset(), w.size(), w.stride())
    }
  }

  def clearParameters[T](model: DeepModel[T]): Unit = {
    val (weightParams, gradParams) = model.parameters()
    weightParams.foreach(_.set())
    gradParams.foreach(_.set())
  }

  def putWeights[T](model: DeepModel[T], newWeights: Seq[Tensor[T]]): Unit = {
    val (weightParams, _) = model.parameters()
    require(weightParams.length == newWeights.length)
    weightParams.zip(newWeights).foreach { case (a, b) => a.set(b) }
  }

  def initGradients[T: ClassTag](model: DeepModel[T], newWeights: Seq[Tensor[T]]): Unit = {
    val (_, gradParams) = model.parameters()
    require(gradParams.length == newWeights.length)
    val storage = Storage[T](gradParams.map(_.nElement()).sum)
    gradParams.zip(newWeights).foreach { case (g, w) =>
      g.set(storage, w.storageOffset(), w.size(), w.stride())
    }
  }
}
