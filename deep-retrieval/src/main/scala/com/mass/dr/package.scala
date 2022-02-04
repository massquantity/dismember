package com.mass

import java.nio.charset.Charset

import com.mass.sparkdl.nn.abstractnn.Activity

package object dr {

  type Path = IndexedSeq[Int]

  type LayerMulti[+T] = Seq[IndexedSeq[T]]

  type LayerModule[T] = com.mass.sparkdl.nn.abstractnn.AbstractModule[Activity, Activity, T]

  type RerankModule[T] = com.mass.sparkdl.nn.abstractnn.AbstractModule[Activity, Activity, T]

  val encoding: Charset = Charset.defaultCharset()

}
