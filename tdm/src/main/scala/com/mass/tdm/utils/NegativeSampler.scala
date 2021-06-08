package com.mass.tdm.utils

import java.util.concurrent.ThreadLocalRandom

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.hashing.MurmurHash3

import com.mass.tdm.tree.DistTree.TreeNode
import com.mass.tdm.tree.TDMTree
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution
import org.apache.commons.math3.random.MersenneTwister

class NegativeSampler(
    tree: TDMTree,
    layerNegCounts: String,
    withProb: Boolean = true,
    startSampleLayer: Int = -1,
    tolerance: Int = 20,
    numThreads: Int) extends Serializable {
  require(tree.initialized, "tree hasn't been initialized...")

  private var negNumPerLayer: Array[Int] = _
  private[tdm] var layerSum: Int = -1
  private var levelProbDistributions: Array[EnumeratedIntegerDistribution] = _
  // private var validCodes: Array[Array[Int]] = _
  // private var validProbs: Array[Array[Double]] = _
  private val totalLevel: Int = tree.maxLevel
  private var isParallel: Boolean = false
  private[tdm] var initialized: Boolean = false

  // def init(): Unit = {
  //  if (withProb) {
  //    validCodes = new Array(totalLevel)
  //    validProbs = new Array(totalLevel)
  //    (0 until totalLevel) foreach { level =>
  //      val (codes, probs) = levelProbs(level)
  //      validCodes(level) = codes
  //      validProbs(level) = probs
  //    }
  //  }
  // }

  def init(): Unit = {
    levelProbDistributions = new Array[EnumeratedIntegerDistribution](totalLevel)
    for (level <- 0 until totalLevel) {
      val generator = new MersenneTwister(System.nanoTime())
      val (codes, probs) = levelProbs(level)
      levelProbDistributions(level) = new EnumeratedIntegerDistribution(generator, codes, probs)
    }
    computeSampleUnit()
    isParallel = false
    initialized = true
  }

  def initParallel(): Unit = {
    levelProbDistributions = new Array[EnumeratedIntegerDistribution](numThreads * totalLevel)
    for (i <- 0 until numThreads) {
      for (level <- 0 until totalLevel) {
        levelProbDistributions(i * totalLevel + level) = {
          val generator = new MersenneTwister(System.nanoTime())
          val (codes, probs) = levelProbs(level)
          new EnumeratedIntegerDistribution(generator, codes, probs)
        }
      }
    }
    computeSampleUnit()
    isParallel = true
    initialized = true
  }

  private def computeSampleUnit(): Unit = {
    val _layerNegCounts = layerNegCounts.split(",")
    require(_layerNegCounts.length >= tree.maxLevel, "Not enough negative sample layers")
    require(_layerNegCounts.zipWithIndex.forall { case (num, i) =>
      num.toInt < math.pow(2, i).toInt
    }, "Num of negative samples must not exceed max numbers in current layer")

    negNumPerLayer = _layerNegCounts.slice(0, tree.maxLevel).map(_.toDouble.toInt)
    // positive(one per layer) + negative nums
    layerSum = negNumPerLayer.length + negNumPerLayer.sum
  }

  private def levelProbs(level: Int): (Array[Int], Array[Double]) = {
    val codes = new ArrayBuffer[Int]()
    val probs = new ArrayBuffer[Double]()
    var index = 0
    (1 to level).foreach(_ => index = index * 2 + 1)
    val end = index * 2 + 1
    while (index < end) {
      if (tree.codeNodeMap.contains(index)) {
        val node = tree.codeNodeMap(index)
        codes += index
        probs += node.probality.toDouble
      }
      index += 1
    }
    require(probs.nonEmpty, s"no probs in level $level")
    (codes.toArray, probs.toArray)
  }

  /**
   *
   * @param itemIds clicked items of one user
   * @param threadId specific thread id
   * @return Tuple2(sampled ids, sampled labels)
   */
  def sample(
      itemIds: Array[Int],
      threadId: Int): (Array[Int], Array[Float]) = {

    val nItems = itemIds.length
    val outputIds = new Array[Int](layerSum * nItems)
    val labels = new Array[Float](layerSum * nItems)
    val itemCodes = tree.idToCode(itemIds)
    val ancestors: Array[Array[TreeNode]] = tree.getAncestorNodes(itemCodes)
    val hasSampled = mutable.HashSet.empty[Int]
    val tid = if (isParallel) threadId else 0

    if (withProb) {
      for (level <- 0 until totalLevel) {
        val seed = NegativeSampler.generateSeed()
        levelProbDistributions(tid * totalLevel + level).reseedRandomGenerator(seed)
      }
    }

    var i = 0
    while (i < ancestors.length) {
      val ancs = ancestors(i)
      var level = totalLevel
      var offset = i * layerSum
      var j = 0
      while (j < ancs.length && level - 1 >= startSampleLayer) {
        level -= 1  // upward sampling, will stop at `startSampleLayer` if possible
        val posNode = ancs(j).node
        val positiveId = posNode.id
        outputIds(offset) = positiveId
        labels(offset) = 1.0f
        offset += 1

        hasSampled.clear()
        val negNum = negNumPerLayer(level)
        if (withProb) {
        //  val seed = NegativeSampler.generateSeed()
        //  val generator = new MersenneTwister(seed)
        //  val weightedDist = new EnumeratedIntegerDistribution(generator,
        //    validCodes(level), validProbs(level))

          val weightedDist = levelProbDistributions(tid * totalLevel + level)
          var t = 0
          val layerTolerance = negNum + tolerance
          // println(s"level $level tolerance: " + layerTolerance)
          while (hasSampled.size < negNum && t < layerTolerance) {
            val s = weightedDist.sample()
            if (!hasSampled.contains(s) && s != positiveId && tree.codeNodeMap.contains(s)) {
              hasSampled += s
            }
            t += 1
          }

          if (t >= layerTolerance) {
            println(s"level $level exceed tolerance, possible cause is " +
              s"popular items with high sampling probabilities")
            val levelStart = (math.pow(2, level) - 1).toInt
            val levelEnd = levelStart * 2 + 1
            val numRemain = negNum - hasSampled.size
            var k = 0
            while (k < numRemain) {
              // try using maxCode EnumeratedIntegerDistribution
              val s = ThreadLocalRandom.current.nextInt(levelStart, levelEnd)
              if (tree.codeNodeMap.contains(s)) {
                hasSampled += s
                k += 1
              }
            }
          }
        } else {
          val levelStart = (math.pow(2, level) - 1).toInt
          val levelEnd = levelStart * 2 + 1
          while (hasSampled.size < negNum) {
            val s = ThreadLocalRandom.current.nextInt(levelStart, levelEnd)
            if (!hasSampled.contains(s) && s != positiveId && tree.codeNodeMap.contains(s)) {
              hasSampled += s
            }
          }
        }

        hasSampled.foreach { s =>
          val negNode = tree.codeNodeMap(s)
          outputIds(offset) = negNode.id
          labels(offset) = 0.0f
          offset += 1
        }
        j += 1
      }
      i += 1
    }
    (outputIds, labels)
  }
}

object NegativeSampler {
  val local = new ThreadLocal[Long]()

  def generateSeed(): Long = {
    val threadHash = MurmurHash3.stringHash(Thread.currentThread.getId.toString)
    // val local = ThreadLocal.withInitial[Long](() => System.nanoTime() + threadHash)
    local.set(System.nanoTime() + threadHash)
    val seed = local.get
    local.remove()
    seed
  }
}
