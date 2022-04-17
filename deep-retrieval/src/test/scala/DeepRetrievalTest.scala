import scala.reflect.runtime.universe.{typeTag, TypeTag}

import com.mass.dr.dataset.LocalDataSet
import com.mass.dr.loss.CrossEntropyLayer
import com.mass.dr.model.{DeepRetrieval, LayerModel, RerankModel}
import com.mass.dr.optim.LocalOptimizer
import com.mass.sparkdl.nn.SampledSoftmaxLoss
import com.mass.sparkdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DeepRetrievalTest extends AnyFlatSpec with Matchers {

  Logger.getLogger("com.mass").setLevel(Level.INFO)
  Engine.setNodeAndCore(1, 8)
  val numLayer = 3
  val numNode = 100
  val numPathPerItem = 2
  val trainBatchSize = 8192
  val evalBatchSize = 8192
  val seqLen = 10
  val minSeqLen = 2
  val dataPath = s"${System.getProperty("user.dir")}/data/data.csv"
  val splitRatio = 0.8
  val paddingIdx = -1
  val embedSize = 16
  val numIteration = 100
  val learningRate = 7e-3
  val numSampled = 1
  val topk = 10
  val beamSize = 20
  val progressInterval = 50

  val dataset = new LocalDataSet(
    numLayer = numLayer,
    numNode = numNode,
    numPathPerItem = numPathPerItem,
    trainBatchSize = trainBatchSize,
    evalBatchSize = evalBatchSize,
    seqLen = seqLen,
    minSeqLen = minSeqLen,
    dataPath = dataPath,
    mappingPath = None,
    splitRatio = splitRatio
  )
  val drModel = DeepRetrieval(
    numItem = dataset.numItem,
    numNode = numNode,
    numLayer = numLayer,
    seqLen = seqLen,
    embedSize = embedSize
  )
  drModel.setMapping(dataset)

  val optimizer = new LocalOptimizer(
    dataset = dataset,
    drModel = drModel,
    numIteration = numIteration,
    numLayer = numLayer,
    learningRate = learningRate,
    numSampled = numSampled,
    embedSize = embedSize,
    topk = topk,
    beamSize = beamSize,
    progressInterval = progressInterval
  )
  optimizer.optimize()

  "DeepRetrieval model" should "initiate with correct classes" in {
    assert(getType(drModel) == "com.mass.dr.model.DeepRetrieval")
    drModel.layerModel shouldBe a [LayerModel]
    drModel.reRankModel shouldBe a [RerankModel]
    optimizer.layerCriterion shouldBe a [CrossEntropyLayer]
    optimizer.reRankCriterion shouldBe a [SampledSoftmaxLoss[_]]
  }

  "Loss in DeepRetrieval model" should "decrease during optimization" in {
    val evalResults = optimizer.optimEvalResults.map(_.meanMetrics)
    evalResults reduce { (former, later) =>
      former.layerLoss lazyZip later.layerLoss foreach((a, b) => assert(a > b))
      assert(former.reRankLoss > later.reRankLoss)
      later
    }
  }

  "DeepRetrieval model" should "return correct recommendations" in {
    val sequence = Array(1, 2, 3, 4, 5, 6, 7, 89, 2628, 1681)
    val recommendResult = drModel.recommend(sequence, 10, 20).map(_._1)
    recommendResult.length should be >= 5
    // recommendResult should contain atLeastOneOf (2628, 2997, 858)
    val totalTime = (1 to 100).map(_ => time(drModel.recommend(sequence, 10, 20)))
    println(f"average recommend time: ${totalTime.sum / totalTime.length}%.4fms")
  }

  def getType[T: TypeTag](obj: T): String = typeTag[T].tpe.toString

  def time[T](block: => T): Double = {
    val start = System.nanoTime()
    val _ = block
    val end = System.nanoTime()
    (end - start) / 1e6d
  }
}
