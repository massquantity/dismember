package com.mass.tdm.utils

import java.util.concurrent.ThreadLocalRandom

import scala.annotation.tailrec
import scala.collection.mutable
import scala.util.hashing.MurmurHash3

import com.mass.tdm.tree.DistTree.TreeNode
import com.mass.tdm.tree.TDMTree
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution
import org.apache.commons.math3.random.MersenneTwister

class NegativeSampler(
    tree: TDMTree,
    layerNegCounts: String,
    withProb: Boolean,
    startSampleLevel: Int,
    tolerance: Int,
    numThreads: Int) extends Serializable {
  require(tree.initialized, "tree hasn't been initialized...")
  require(startSampleLevel > 0, s"start sample level should be at least 1, got $startSampleLevel")

  private var negNumPerLayer: Array[Int] = _
  private[tdm] var layerSum: Int = -1
  private var levelProbDistributions: IndexedSeq[EnumeratedIntegerDistribution] = _
  private val totalLevel: Int = tree.maxLevel + 1
  private[tdm] var initialized: Boolean = false

  def initParallel(): this.type = {
    if (withProb) {
      levelProbDistributions =
        for {
          _ <- 0 until numThreads
          level <- 0 until totalLevel
          generator = new MersenneTwister(System.nanoTime())
          (codes, probs) = levelProbs(level)
        } yield new EnumeratedIntegerDistribution(generator, codes, probs)
    }
    computeSampleUnit()
    initialized = true
    this
  }

  private def computeSampleUnit(): Unit = {
    val _layerNegCounts = layerNegCounts.split(",")
    require(_layerNegCounts.length >= totalLevel, "Not enough negative sample layers")
    require(_layerNegCounts.zipWithIndex.forall { case (num, i) =>
      num.toInt < math.pow(2, i).toInt
    }, "Num of negative samples must not exceed max numbers in current layer")

    negNumPerLayer = _layerNegCounts.take(totalLevel).map(_.toDouble.toInt)
    // positive(one per layer) + negative nums, exclude root node
    layerSum = negNumPerLayer.length - 1 + negNumPerLayer.sum - negNumPerLayer.head
  }

  private def levelProbs(level: Int): (Array[Int], Array[Double]) = {
    val start = NegativeSampler.getStartNode(level, 0)
    val end = start * 2 + 1
    val codes = (start until end).toArray.filter(tree.codeNodeMap.contains)
    val probs = codes.map(tree.codeNodeMap(_).probality.toDouble)
    require(probs.nonEmpty, s"no probs in level $level")
    (codes, probs)
  }

  /**
   *
   * @param itemIds consumed items of one user
   * @param threadId specific thread id
   * @return Tuple2(sampled ids, sampled labels)
   */
  def sample(itemIds: Seq[Int], threadId: Int): (Seq[Int], Seq[Float]) = {
    val (itemCodes, _) = tree.idToCode(itemIds.toArray)
    val allPathNodes: Seq[List[TreeNode]] = tree.pathNodes(itemCodes)
    if (withProb) {
      for (level <- 0 until totalLevel) {
        val seed = NegativeSampler.generateSeed()
        levelProbDistributions(threadId * totalLevel + level).reseedRandomGenerator(seed)
      }
    }

    val sampledCodes =
      for {
        pathNodes <- allPathNodes
        // The sampling should start from at least level 1, but relative indices of pathNodes start from 0.
        // Also the left side values of pathNodes are close to the root node, so those are dropped first.
        validNodes = startSampleLevel match {
          case 1 => pathNodes
          case n => pathNodes.drop(n - 1)
        }
        validLevels = startSampleLevel until totalLevel
        (node, level) <- validNodes zip validLevels
        posCode = node.code
        negCodes =
          if (withProb) {
            sampleFromCategoricalDistribution(level, posCode, threadId)
          } else {
            sampleFromUniformDistribution(level, posCode)
          }
      } yield posCode :: negCodes

    val labels =
      for {
        _ <- allPathNodes
        level <- startSampleLevel until totalLevel
        negNum = negNumPerLayer(level)
      } yield 1.0f :: List.fill(negNum)(0.0f)

    (sampledCodes.flatten, labels.flatten)
  }

  def sampleFromCategoricalDistribution(level: Int, posCode: Int, threadId: Int): List[Int] = {
    val hasSampled = mutable.BitSet.empty
    val weightedDist = levelProbDistributions(threadId * totalLevel + level)
    val negNum = negNumPerLayer(level)
    val layerTolerance = negNum + tolerance
    var t = 0
    while (hasSampled.size < negNum && t < layerTolerance) {
      val s = weightedDist.sample()
      if (!hasSampled.contains(s) && s != posCode && tree.codeNodeMap.contains(s)) {
        hasSampled += s
      }
      t += 1
    }
    if (hasSampled.size < negNum) {
      println(s"level $level exceed tolerance, possible cause is " +
        s"popular items with high sampling probabilities")
      val levelStart = (math.pow(2, level) - 1).toInt
      val levelEnd = levelStart * 2 + 1
      while (hasSampled.size < negNum) {
        val s = ThreadLocalRandom.current.nextInt(levelStart, levelEnd)
        if (tree.codeNodeMap.contains(s)) {
          hasSampled += s
        }
      }
    }
    hasSampled.toList
  }

  def sampleFromUniformDistribution(level: Int, posCode: Int): List[Int] = {
    val hasSampled = mutable.BitSet.empty
    val negNum = negNumPerLayer(level)
    val levelStart = (math.pow(2, level) - 1).toInt
    val levelEnd = levelStart * 2 + 1
    while (hasSampled.size < negNum) {
      val s = ThreadLocalRandom.current.nextInt(levelStart, levelEnd)
      if (!hasSampled.contains(s) && s != posCode && tree.codeNodeMap.contains(s)) {
        hasSampled += s
      }
    }
    hasSampled.toList
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

  // val start = (1 to level).foldLeft(0)((i, _) => i * 2 + 1)
  @tailrec
  def getStartNode(level: Int, n: Int): Int = {
    if (level == 0) n
    else getStartNode(level - 1, n * 2 + 1)
  }
}
