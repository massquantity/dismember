package com.mass.scalann.tensor

import scala.language.implicitConversions

trait ConvertableTo[@specialized A] {
  implicit def fromFloat(a: Float): A

  implicit def fromDouble(a: Double): A

  implicit def fromInt(a: Int): A

  implicit def fromLong(a: Long): A
}

trait ConvertableToLong extends ConvertableTo[Long] {
  implicit def fromFloat(a: Float): Long = a.toLong

  implicit def fromDouble(a: Double): Long = a.toLong

  implicit def fromInt(a: Int): Long = a.toLong

  implicit def fromLong(a: Long): Long = a
}

trait ConvertableToFloat extends ConvertableTo[Float] {
  implicit def fromFloat(a: Float): Float = a

  implicit def fromDouble(a: Double): Float = a.toFloat

  implicit def fromInt(a: Int): Float = a.toFloat

  implicit def fromLong(a: Long): Float = a.toFloat
}

trait ConvertableToDouble extends ConvertableTo[Double] {
  implicit def fromFloat(a: Float): Double = a.toDouble

  implicit def fromDouble(a: Double): Double = a

  implicit def fromInt(a: Int): Double = a.toDouble

  implicit def fromLong(a: Long): Double = a.toDouble
}

trait ConvertableToInt extends ConvertableTo[Int] {
  implicit def fromFloat(a: Float): Int = a.toInt

  implicit def fromDouble(a: Double): Int = a.toInt

  implicit def fromInt(a: Int): Int = a

  implicit def fromLong(a: Long): Int = a.toInt
}

object ConvertableTo {
  implicit object ConvertableToFloat extends ConvertableToFloat

  implicit object ConvertableToDouble extends ConvertableToDouble

  implicit object ConvertableToInt extends ConvertableToInt

  implicit object ConvertableToLong extends ConvertableToLong
}


trait ConvertableFrom[@specialized A] {
  implicit def toFloat(a: A): Float

  implicit def toDouble(a: A): Double

  implicit def toLong(a: A): Long

  implicit def toInt(a: A): Int
}

trait ConvertableFromFloat extends ConvertableFrom[Float] {
  implicit def toFloat(a: Float): Float = a

  implicit def toDouble(a: Float): Double = a.toDouble

  implicit def toInt(a: Float): Int = a.toInt

  implicit def toLong(a: Float): Long = a.toLong

  implicit def toString(a: Float): String = a.toString
}

trait ConvertableFromDouble extends ConvertableFrom[Double] {
  implicit def toFloat(a: Double): Float = a.toFloat

  implicit def toDouble(a: Double): Double = a

  implicit def toInt(a: Double): Int = a.toInt

  implicit def toLong(a: Double): Long = a.toLong

  implicit def toString(a: Double): String = a.toString
}

trait ConvertableFromInt extends ConvertableFrom[Int] {
  implicit def toFloat(a: Int): Float = a.toFloat

  implicit def toDouble(a: Int): Double = a.toDouble

  implicit def toInt(a: Int): Int = a

  implicit def toLong(a: Int): Long = a.toLong

  implicit def toString(a: Int): String = a.toString
}

trait ConvertableFromLong extends ConvertableFrom[Long] {
  implicit def toFloat(a: Long): Float = a.toFloat

  implicit def toDouble(a: Long): Double = a.toDouble

  implicit def toInt(a: Long): Int = a.toInt

  implicit def toLong(a: Long): Long = a

  implicit def toString(a: Long): String = a.toString
}

object ConvertableFrom {
  implicit object ConvertableFromFloat extends ConvertableFromFloat

  implicit object ConvertableFromDouble extends ConvertableFromDouble

  implicit object ConvertableFromInt extends ConvertableFromInt

  implicit object ConvertableFromLong extends ConvertableFromLong
}
