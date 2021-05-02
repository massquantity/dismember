package org.apache.spark.sparkExtension

import org.apache.spark.storage.{BlockId, TestBlockId}

object SparkExtension {
  def getLocalBlockId(id: String): BlockId = {
    TestBlockId(id)
  }
}
