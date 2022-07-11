package com.mass

import java.nio.charset.Charset

import com.mass.sparkdl.nn.abstractnn.Activity

package object otm {

  val paddingIdx: Int = -1

  val encoding: Charset = Charset.defaultCharset()

  type DeepModel[T] = com.mass.sparkdl.nn.abstractnn.AbstractModule[Activity, Activity, T]

  val lowerLog2 = (n: Int) => math.floor(math.log(n) / math.log(2)).toInt

  val upperLog2 = (n: Int) => math.ceil(math.log(n) / math.log(2)).toInt

  val clipValue = (value: Double, min: Double, max: Double) => math.max(min, math.min(max, value))

}
