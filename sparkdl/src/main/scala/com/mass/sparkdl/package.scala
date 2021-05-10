package com.mass

import scala.language.implicitConversions

import com.mass.sparkdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.mass.sparkdl.tensor.Tensor

package object sparkdl {

  type Module[T] = com.mass.sparkdl.nn.abstractnn.AbstractModule[Activity, Activity, T]

  type Criterion[T] = com.mass.sparkdl.nn.abstractnn.AbstractCriterion[T]

  type DataSet[D] = com.mass.sparkdl.dataset.AbstractDataSet[D, _]

  implicit def convertModule[T](module: AbstractModule[_, _, T]): Module[T] = {
    module.asInstanceOf[Module[T]]
  }

  def getScalaVersion: String = scala.util.Properties.versionNumberString

  def getSparkVersion: String = org.apache.spark.SPARK_VERSION
}
