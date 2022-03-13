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
      val (leftIndex, rightIndex) = cluster(index, embeddings, clusterIterNum, clusterType)
      ForkJoinTask.invokeAll(
        new ForkJoinProcess(
          2 * pcode + 1,
          leftIndex,
          codes,
          embeddings,
          threshold,
          clusterIterNum,
          clusterType
        ),
        new ForkJoinProcess(
          2 * pcode + 2,
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
