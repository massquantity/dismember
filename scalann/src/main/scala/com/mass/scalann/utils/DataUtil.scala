package com.mass.scalann.utils

import org.apache.commons.math3.random.RandomDataGenerator

object DataUtil {

  def shuffle[T](data: Array[T]): Array[T] = {
    var i = 0
    val length = data.length
    val generator = new RandomDataGenerator()
    while (i < length) {
      val exchange = generator.nextUniform(0, length - i).toInt + i
      val tmp = data(exchange)
      data(exchange) = data(i)
      data(i) = tmp
      i += 1
    }
    data
  }
}
