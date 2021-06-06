package com.mass.tdm.cluster

import java.io._
import java.util.concurrent.ForkJoinPool

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.{Failure, Success, Try, Using}

import com.mass.sparkdl.utils.{FileReader => DistFileReader}
import com.mass.tdm.ArrayExtension
import com.mass.tdm.tree.TreeBuilder
import org.apache.log4j.{Level, Logger}
import smile.clustering.{KMeans, PartitionClustering}

// todo: first spectral clustering, then Kmeans
class RecursiveCluster(
    parallel: Boolean = false,
    numThreads: Int = 1,
    delimiter: String = ",",
    threshold: Int = 256,
    clusterIterNum: Int = 10) {

  import RecursiveCluster._
  Logger.getLogger("smile").setLevel(Level.ERROR)

  private var ids: Array[Int] = _
  private var codes: Array[Int] = _
  private var embeddings: Array[Array[Double]] = _
  private var stat: Option[mutable.Map[Int, Int]] = None
  require(threshold >= 4, "threshold should be no less than 4")

  def getCodes: Array[Int] = codes

  def getIds: Array[Int] = ids

  def run(
      embedPath: String,
      outputTreePath: String,
      statPath: Option[String] = None): Unit = {

    readFile(embedPath, statPath)

    if (parallel) {
      trainParallel(0, ids.indices.toArray)
    } else {
      train(0, ids.indices.toArray)
    }

    val builder = new TreeBuilder(outputTreePath)
    builder.build(treeIds = ids, treeCodes = codes, stat = stat)
  }

  def readFile(embedPath: String, statPath: Option[String]): Unit = {
    val _ids = new ArrayBuffer[Int]()
    val _embeddings = new ArrayBuffer[Array[Double]]()

    val fileReader = DistFileReader(embedPath)
    val inputStream = fileReader.open()
    Using(new BufferedReader(new InputStreamReader(inputStream))) { input =>
      Iterator.continually(input.readLine()).takeWhile(_ != null).foreach { line =>
        val arr = line.trim.split(delimiter)
        _ids += arr.head.trim.toDouble.toInt
        _embeddings += arr.tail.map(_.trim.toDouble)
      }
    } match {
      case Success(_) =>
        inputStream.close()
        fileReader.close()
      case Failure(e: FileNotFoundException) =>
        println(s"""file "$embedPath" not found""")
        throw e
      case Failure(t: Throwable) =>
        throw t
    }

    ids = _ids.toArray
    codes = Array.fill[Int](ids.length)(0)
    embeddings = _embeddings.toArray

    statPath match {
      case Some(sf) =>
        val _stat = mutable.HashMap.empty[Int, Int]
        val fileReader = DistFileReader(sf)
        val inputStream = fileReader.open()
        Using(new BufferedReader(new InputStreamReader(inputStream))) { sfInput =>
          sfInput.lines.forEach { line =>
            val arr = line.split(delimiter)
            _stat(arr(0).toDouble.toInt) = arr(1).toDouble.toInt
          }
        } match {
          case Success(_) =>
          case Failure(e: Throwable) =>
            throw e
        }
        stat = Some(_stat)
      case None =>
    }
  }

  def train(pcode: Int, index: Array[Int]): Unit = {
    if (index.length <= threshold) {
      minibatch(pcode, index, codes, embeddings, clusterIterNum)
    } else {
      val (leftIndex, rightIndex) = cluster(index, embeddings, clusterIterNum)
      train(2 * pcode + 1, leftIndex)
      train(2 * pcode + 2, rightIndex)
    }
  }

  def trainParallel(pcode: Int, index: Array[Int]): Unit = {
    // val pool = ForkJoinPool.commonPool()
    val pool = new ForkJoinPool(numThreads)
    // println("parallelism: " + pool.getParallelism + " " + pool.getPoolSize)
    val task = new ForkJoinProcess(pcode, index, codes, embeddings, threshold, clusterIterNum)
    pool.invoke(task)
  }
}

object RecursiveCluster {

  def minibatch(
      pcode: Int,
      index: Array[Int],
      codes: Array[Int],
      embeddings: Array[Array[Double]],
      ClusterIterNum: Int): Unit = {

    val queue = mutable.Queue.empty[(Int, Array[Int])]
    queue += Tuple2(pcode, index)

    while (queue.nonEmpty) {
      val (code, idx) = queue.dequeue()
      val (leftCode, rightCode) = (2 * code + 1, 2 * code + 2)
      if (idx.length == 2) {
        codes(idx(0)) = leftCode
        codes(idx(1)) = rightCode
      } else {
        val (leftIndex, rightIndex) = cluster(idx, embeddings, ClusterIterNum)
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

  def cluster(
      index: Array[Int],
      embeddings: Array[Array[Double]],
      ClusterIterNum: Int): (Array[Int], Array[Int]) = {

    val embedPartial = index.map(i => embeddings(i))
    val kmeansModel: KMeans = PartitionClustering.run(
      ClusterIterNum, () => KMeans.fit(embedPartial, 2))
    // choose one centroid to compute and sort according to distance,
    // then split into two subsets
    val centroid = kmeansModel.centroids.head
    val distance = embedPartial.map(emb => squaredDistance(emb, centroid))
    balanceTree(distance, index)
  }

  def balanceTree(distance: Array[Double], index: Array[Int]): (Array[Int], Array[Int]) = {
  //  val mid = distance.length / 2
  //  val leftIndex = new Array[Int](mid)
  //  val rightIndex = new Array[Int](distance.length - mid)
  //  val indirectIdx = distance.argPartition(inplace = true, position = mid)
  //  var i = 0
  //  val len = distance.length
  //  while (i < len) {
  //    if (i < mid) {
  //      leftIndex(i) = index(indirectIdx(i))
  //    } else {
  //      rightIndex(i - mid) = index(indirectIdx(i))
  //    }
  //    i += 1
  //  }
  //  (leftIndex, rightIndex)

    val mid = distance.length / 2
    val indirectIdx = distance.argPartition(inplace = true, position = mid).splitAt(mid)
    (indirectIdx._1.map(index(_)), indirectIdx._2.map(index(_)))
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
