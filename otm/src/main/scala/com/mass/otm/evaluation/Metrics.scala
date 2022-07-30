package com.mass.otm.evaluation

import com.mass.otm.tree.OTMTree.Node

object Metrics {

  def computeMetrics(recItems: Seq[Node], labels: Seq[Int]): EvalResult = {
    val labelSet = labels.toSet
    var j, commonItems = 0
    var dcg, idcg = 0.0
    recItems.zipWithIndex.foreach { case (item, i) =>
      if (labelSet.contains(item.id)) {
        commonItems += 1
        dcg += (math.log(2) / math.log(i + 2))
        idcg += (math.log(2) / math.log(j + 2))
        j += 1
      }
    }

    if (commonItems != 0) {
      EvalResult(
        commonItems.toDouble / recItems.length,
        commonItems.toDouble / labels.length,
        dcg / idcg
      )
    } else {
      EvalResult(0.0, 0.0, 0.0)
    }
  }
}
