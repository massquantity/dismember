package com.mass

import java.nio.charset.Charset

import com.mass.sparkdl.nn.abstractnn.Activity

package object otm {

  val encoding: Charset = Charset.defaultCharset()

  type DeepModel[T] = com.mass.sparkdl.nn.abstractnn.AbstractModule[Activity, Activity, T]

}
