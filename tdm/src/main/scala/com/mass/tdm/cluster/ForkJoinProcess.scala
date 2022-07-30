package com.mass.tdm.cluster

import java.util.concurrent.{ForkJoinTask, RecursiveAction}

import RecursiveCluster.{miniBatch, cluster}

class ForkJoinProcess(
    pcode: Int,
    index: Array[Int],
    codes: Array[Int],
    embeddings: Array[Array[Double]],
    threshold: Int,
    clusterIterNum: Int,
    clusterType: String) extends RecursiveAction {

  override def compute(): Unit = {
    if (index.length <= threshold) {
      miniBatch(pcode, index, codes, embeddings, clusterIterNum, clusterType)
    } else {
      val (leftCode, rightCode) = (2 * pcode + 1, 2 * pcode + 2)
      val (leftIndex, rightIndex) = cluster(index, embeddings, clusterIterNum, clusterType)
      ForkJoinTask.invokeAll(
        new ForkJoinProcess(
          leftCode,
          leftIndex,
          codes,
          embeddings,
          threshold,
          clusterIterNum,
          clusterType
        ),
        new ForkJoinProcess(
          rightCode,
          rightIndex,
          codes,
          embeddings,
          threshold,
          clusterIterNum,
          clusterType
        )
      )
    }
  }
}
