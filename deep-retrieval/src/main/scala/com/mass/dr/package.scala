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

  val sigmoid = (logit: Double) => 1.0 / (1 + java.lang.Math.exp(-logit))

  val softmax = (logits: Seq[Double]) => {
    val maxVal = logits.max
    val exps = logits.map(i => java.lang.Math.exp(i - maxVal)) // avoid overflow in exp
    val sumExps = exps.sum
    exps.map(_ / sumExps)
  }
}
