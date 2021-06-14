package com.mass

import java.nio.charset.Charset

import scala.reflect.ClassTag

import com.mass.tdm.utils.Utils

package object tdm {

  val encoding: Charset = Charset.defaultCharset()

  case class Feature(itemSeqs: Array[Int], masks: Array[Int] = null)

  implicit class ArrayExtension[@specialized(Int, Long, Float, Double) T](array: Array[T])(
      implicit order: T => Ordered[T]) {

    def argSort(inplace: Boolean): Array[Int] = {
      val arr = if (inplace) array else array.clone()
      val indices = arr.indices.toArray
      Utils.argSort(arr, indices)
      indices
    }

    def argPartition(inplace: Boolean, position: Int): Array[Int] = {
      val arr = if (inplace) array else array.clone()
      val indices = arr.indices.toArray
      Utils.argPartition(arr, position, indices)
      indices
    }
  }

  def getType[T](v: T)(implicit ev: ClassTag[T]): String = ev.toString

  def getScalaVersion: String = scala.util.Properties.versionNumberString

  def getSparkVersion: String = org.apache.spark.SPARK_VERSION
}
