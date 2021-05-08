package com.mass.sparkdl.dataset

import com.mass.sparkdl.utils.Engine
import org.apache.commons.math3.random.RandomDataGenerator
import org.apache.log4j.Logger

object DataUtil {
  private val logger = Logger.getLogger(getClass)

  def getBatchSize(batchSize: Int, totalPartition: Option[Int] = None): Int = {
    val nodeNumber = Engine.nodeNumber()
    val partitionNum = totalPartition.getOrElse(nodeNumber)
    logger.debug(s"partition number: $partitionNum, node number: $nodeNumber")
    require(batchSize % partitionNum == 0, "total batchSize should be divided be partitionNum")

    val batchPerUnit = batchSize / partitionNum
    logger.debug(s"Batch per unit: $batchPerUnit")
    batchPerUnit
  }

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
