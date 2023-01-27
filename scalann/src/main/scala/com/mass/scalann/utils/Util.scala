package com.mass.scalann.utils

import scala.annotation.tailrec
import scala.reflect.ClassTag

import com.mass.scalann.tensor.{Storage, Tensor, TensorNumeric}
import com.mass.scalann.Module

object Util {

  @tailrec
  def kthLargest(arr: Array[Long], lo: Int, hi: Int, k: Int): Long = {
    if (k == 0) return Long.MaxValue
    val pos = randomPartition(arr, lo, hi)
    if (pos - lo == k - 1) {
      arr(pos)
    } else if (pos - lo > k - 1) {
      kthLargest(arr, lo, pos - 1, k)
    } else {
      kthLargest(arr, pos + 1, hi, k - pos + lo - 1)
    }
  }

  def swap(arr: Array[Long], i: Int, j: Int): Unit = {
    val temp = arr(i)
    arr(i) = arr(j)
    arr(j) = temp
  }

  private def partition(arr: Array[Long], lo: Int, hi: Int): Int = {
    val x = arr(hi)
    var i = lo
    var j = lo
    while (j < hi) {
      if (arr(j) > x) {
        swap(arr, i, j)
        i += 1
      }
      j += 1
    }
    swap(arr, i, hi)
    i
  }

  private def randomPartition(arr: Array[Long], lo: Int, hi: Int): Int = {
    val n = hi - lo + 1
    val pivot = (Math.random() % n).toInt
    swap(arr, lo + pivot, hi)
    partition(arr, lo, hi)
  }

  def getAndClearWeightBias[T: ClassTag](
      parameters: (Array[Tensor[T]], Array[Tensor[T]])
  )(implicit ev: TensorNumeric[T]): Array[Tensor[T]] = {
    if (parameters._1.nonEmpty) {
      val weightBias = new Array[Tensor[T]](parameters._1.length)
      val firstStorage = Storage(parameters._1.head.storage().array())
      val isCompacted = parameters._1.map(_.nElement()).sum == firstStorage.length
      parameters._1.zipWithIndex.foreach { case (wb, i) =>
        if (wb != null) {
          weightBias(i) = {
            if (isCompacted) {
              Tensor[T](firstStorage, wb.storageOffset(), wb.size(), wb.stride())
            } else {
              Tensor[T](Storage(wb.storage().array()), wb.storageOffset(), wb.size(), wb.stride())
            }
          }
        }
      }

      clearTensor(parameters._1)
      clearTensor(parameters._2)
      weightBias
    } else {
      Array.empty
    }
  }

  def clearTensor[T: ClassTag](tensors: Array[Tensor[T]])(implicit ev: TensorNumeric[T]): Unit = {
    var i = 0
    while (i < tensors.length) {
      if (tensors(i) != null) {
        tensors(i).set()
      }
      i += 1
    }
  }

  def putWeightBias[T: ClassTag](broadcastWeightBias: Array[Tensor[T]], localModel: Module[T])(
      implicit ev: TensorNumeric[T]
  ): Unit = {
    val localWeightBias = localModel.parameters()._1
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias != null) {
        localWeightBias(i).set(broadcastWeightBias(i))
      }
      i += 1
    }
  }

  def initGradWeightBias[T: ClassTag](broadcastWeightBias: Array[Tensor[T]], localModel: Module[T])(
      implicit ev: TensorNumeric[T]
  ): Unit = {
    val (localWeightBias, localGradWeightBias) = localModel.parameters()
    // init gradient with a compacted storage
    val storage = Storage[T](localGradWeightBias.map(_.nElement()).sum)
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        val wb = broadcastWeightBias(i)
        localGradWeightBias(i).set(storage, wb.storageOffset(), wb.size(), wb.stride())
      }
      i += 1
    }
  }

  def cloneParameters[T: ClassTag](
      parameters: Array[Tensor[T]]
  )(implicit ev: TensorNumeric[T]): Array[Tensor[T]] = {
    if (parameters == null) return null
    if (parameters.isEmpty) return Array.empty

    val retParams = new Array[Tensor[T]](parameters.length)
    val first = parameters(0)
    val length = first.storage().length
    val isCompacted = parameters.map(_.nElement()).sum == length

    val resultStorage = {
      if (isCompacted) {
        val tmp = Storage[T](length)
        System.arraycopy(first.storage().array(), first.storageOffset(), tmp.array(), 0, length)
        tmp
      } else {
        null
      }
    }

    var i = 0
    while (i < parameters.length) {
      if (parameters(i) != null) {
        val param = parameters(i)
        retParams(i) = {
          if (isCompacted) {
            Tensor[T](resultStorage, param.storageOffset(), param.size(), param.stride())
          } else {
            param.clone()
          }
        }
      }
      i += 1
    }
    retParams
  }
}
