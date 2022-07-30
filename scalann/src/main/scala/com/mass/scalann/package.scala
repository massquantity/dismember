package com.mass

import scala.language.implicitConversions

import com.mass.scalann.nn.abstractnn.{AbstractModule, Activity}

package object scalann {

  type Module[T] = com.mass.scalann.nn.abstractnn.AbstractModule[Activity, Activity, T]

  type Criterion[T] = com.mass.scalann.nn.abstractnn.AbstractCriterion[T]

  implicit def convertModule[T](module: AbstractModule[_, _, T]): Module[T] = {
    module.asInstanceOf[Module[T]]
  }

  def getScalaVersion: String = scala.util.Properties.versionNumberString

}
