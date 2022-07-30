package com.mass

import java.nio.charset.Charset

import com.mass.scalann.nn.abstractnn.Activity

package object dr {

  type Path = IndexedSeq[Int]

  type LayerMulti[+T] = Seq[IndexedSeq[T]]

  type LayerModule[T] = com.mass.scalann.nn.abstractnn.AbstractModule[Activity, Activity, T]

  type RerankModule[T] = com.mass.scalann.nn.abstractnn.AbstractModule[Activity, Activity, T]

  val encoding: Charset = Charset.defaultCharset()

  val paddingIdx: Int = -1

}
