package com.mass.tdm.cluster

import java.util.concurrent.{ForkJoinTask, RecursiveAction}

import RecursiveCluster.{minibatch, cluster}

class ForkJoinProcess(
    pcode: Int,
    index: Array[Int],
    codes: Array[Int],
    embeddings: Array[Array[Double]],
    threshold: Int,
    clusterIterNum: Int) extends RecursiveAction {

  override def compute(): Unit = {
    if (index.length <= threshold) {
      minibatch(pcode, index, codes, embeddings, clusterIterNum)
    } else {
      val (leftIndex, rightIndex) = cluster(index, embeddings, clusterIterNum)
      ForkJoinTask.invokeAll(
        new ForkJoinProcess(2 * pcode + 1, leftIndex, codes, embeddings, threshold, clusterIterNum),
        new ForkJoinProcess(2 * pcode + 2, rightIndex, codes, embeddings, threshold, clusterIterNum)
      )
    }
  }
}
