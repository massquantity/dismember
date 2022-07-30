package com.mass.dr.evaluation

object Metrics {

  def computeMetrics(recItems: Seq[Int], labels: Seq[Int]): (Double, Double, Double) = {
    val k = recItems.length
    val labelSet = labels.toSet
    var j, commonItems = 0
    var dcg, idcg = 0.0
    recItems.zipWithIndex.foreach { case (item, i) =>
      if (labelSet.contains(item)) {
        commonItems += 1
        dcg += (java.lang.Math.log(2) / java.lang.Math.log(i + 2))
        idcg += (java.lang.Math.log(2) / java.lang.Math.log(j + 2))
        j += 1
      }
    }

    if (commonItems != 0) {
      (commonItems.toDouble / k, commonItems.toDouble / labels.length, dcg / idcg)
    } else {
      (0.0, 0.0, 0.0)
    }
  }
}
