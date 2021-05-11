package com.mass.tdm.evaluation

object Metrics {

  def computeMetrics(recItems: Array[Int], labels: Array[Int]): (Double, Double, Double) = {
    val k = recItems.length
    val labelSet = labels.toSet
    var i, j, commonItems = 0
    var dcg, idcg = 0.0
    while (i < k) {
      if (labelSet.contains(recItems(i))) {
        commonItems += 1
        dcg += (java.lang.Math.log(2) / java.lang.Math.log(i + 2))
        idcg += (java.lang.Math.log(2) / java.lang.Math.log(j + 2))
        j += 1
      }
      i += 1
    }
  //  val precision = if (commonItems != 0) commonItems.toDouble / k else 0.0
  //  val recall = if (commonItems != 0) commonItems.toDouble / labels.length else 0.0
  //  val ndcg = if (commonItems != 0) dcg / idcg else 0.0
  //  (precision, recall, ndcg)

    if (commonItems != 0) {
      (commonItems.toDouble / k, commonItems.toDouble / labels.length, dcg / idcg)
    } else {
      (0.0, 0.0, 0.0)
    }
  }

}
