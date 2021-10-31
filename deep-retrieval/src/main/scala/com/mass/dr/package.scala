package com.mass

import java.nio.charset.Charset

package object dr {

  type Path = IndexedSeq[Int]

  val encoding: Charset = Charset.defaultCharset()

}
