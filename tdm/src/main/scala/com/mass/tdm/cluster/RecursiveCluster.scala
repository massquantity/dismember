package com.mass.tdm.cluster

import java.io._
import java.util.concurrent.ForkJoinPool

import scala.collection.mutable
import scala.util.Using

import com.mass.clustering.SpectralClustering
import com.mass.scalann.utils.{FileReader => DistFileReader}
import com.mass.tdm.ArrayExtension
import com.mass.tdm.tree.TreeBuilder
import org.apache.log4j.{Level, Logger}
import smile.clustering.{KMeans, PartitionClustering}

class RecursiveCluster(
    embedPath: String,
    outputTreePath: String,
    parallel: Boolean,
    numThreads: Int,
    delimiter: String = ",",
    threshold: Int = 256,
    clusterIterNum: Int = 10,
    clusterType: String = "kmeans") {
  import RecursiveCluster._
  Logger.getLogger("smile").setLevel(Level.ERROR)
  require(threshold >= 4, "threshold should be no less than 4")
  require(clusterType == "kmeans" || clusterType == "spectral",
    s"clusterType must be one of ('kmeans', 'spectral')")
  if (clusterType == "spectral") {
    require(!parallel, "spectral clustering does not support parallel mode.")
  }

  val (ids, embeddings): (Array[Int], Array[Array[Double]]) = readFile(embedPath, delimiter)
  val codes: Array[Int] = Array.fill[Int](ids.length)(0)

  def run(): Unit = {
    if (parallel) {
      trainParallel(0, ids.indices.toArray)
    } else {
      train(0, ids.indices.toArray)
    }

    TreeBuilder.build(
      outputTreePath = outputTreePath,
      treeIds = ids,
      treeCodes = codes
    )
  }

  def train(pcode: Int, index: Array[Int]): Unit = {
    if (index.length <= threshold) {
      miniBatch(pcode, index, codes, embeddings, clusterIterNum, clusterType)
    } else {
      val (leftCode, rightCode) = (2 * pcode + 1, 2 * pcode + 2)
      val (leftIndex, rightIndex) = cluster(index, embeddings, clusterIterNum, clusterType)
      train(leftCode, leftIndex)
      train(rightCode, rightIndex)
    }
  }

  def trainParallel(pcode: Int, index: Array[Int]): Unit = {
    // val pool = ForkJoinPool.commonPool()
    val pool = new ForkJoinPool(numThreads)
    val task = new ForkJoinProcess(
      pcode,
      index,
      codes,
      embeddings,
      threshold,
      clusterIterNum,
      clusterType
    )
    pool.invoke(task)
    pool.shutdown()
  }
}

object RecursiveCluster {

  def readFile(embedPath: String, delimiter: String): (Array[Int], Array[Array[Double]]) = {
    val fileReader = DistFileReader(embedPath)
    val inputStream = fileReader.open()
    val lines = Using.resource(new BufferedReader(new InputStreamReader(inputStream))) { reader =>
      Iterator.continually(reader.readLine()).takeWhile(_ != null).toArray
    }.map(_.split(delimiter))
    val ids = lines.map(_.head.trim.toInt)
    val embeds = lines.map(_.tail.map(_.trim.toDouble))
    (ids, embeds)
  }

  def miniBatch(
    pcode: Int,
    index: Array[Int],
    codes: Array[Int],
    embeddings: Array[Array[Double]],
    clusterIterNum: Int,
    clusterType: String
  ): Unit = {
    val queue = mutable.Queue[(Int, Array[Int])](Tuple2(pcode, index))
    while (queue.nonEmpty) {
      val (code, idx) = queue.dequeue()
      val (leftCode, rightCode) = (2 * code + 1, 2 * code + 2)
      if (idx.length == 2) {
        codes(idx(0)) = leftCode
        codes(idx(1)) = rightCode
      } else {
        val (leftIndex, rightIndex) = cluster(idx, embeddings, clusterIterNum, clusterType)
        if (leftIndex.length == 1) {
          codes(leftIndex.head) = leftCode
        } else {
          queue += Tuple2(leftCode, leftIndex)
        }
        if (rightIndex.length == 1) {
          codes(rightIndex.head) = rightCode
        } else {
          queue += Tuple2(rightCode, rightIndex)
        }
      }
    }
  }

  // choose one centroid to compute and sort according to distance,
  // then split into two subsets.
  def cluster(
    index: Array[Int],
    embeddings: Array[Array[Double]],
    clusterIterNum: Int,
    clusterType: String
  ): (Array[Int], Array[Int]) = {
    val embedPartial = index.map(embeddings(_))
    val (centroid, matrix) = clusterType match {
      case "kmeans" =>
        val kmeansModel: KMeans = PartitionClustering.run(
          clusterIterNum, () => KMeans.fit(embedPartial, 2))
        (kmeansModel.centroids.head, embedPartial)
      case "spectral" =>
        val clusterResult = SpectralClustering.fit(embedPartial, 2, 1.0, clusterIterNum)
        (clusterResult.getLeft, clusterResult.getRight)
    }
    val distance = matrix.map(emb => squaredDistance(emb, centroid))
    balanceTree(distance, index)
  }

  def balanceTree(distance: Array[Double], index: Array[Int]): (Array[Int], Array[Int]) = {
    val mid = distance.length / 2
    val (leftPart, rightPart) = distance.argPartition(mid, inplace = true).splitAt(mid)
    (leftPart.map(index(_)), rightPart.map(index(_)))
  }

  @inline
  def squaredDistance(x: Array[Double], y: Array[Double]): Double = {
    var sum = 0.0
    var i = 0
    while (i < x.length) {
      val d = x(i) - y(i)
      sum += d * d
      i += 1
    }
    sum
  }
}
