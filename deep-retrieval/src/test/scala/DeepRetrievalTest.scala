import scala.reflect.runtime.universe.{typeTag, TypeTag}

import com.mass.dr.dataset.LocalDataSet
import com.mass.dr.loss.CrossEntropyLayer
import com.mass.dr.model.{DeepRetrieval, LayerModel, MappingOp, RerankModel}
import com.mass.dr.optim.LocalOptimizer
import com.mass.scalann.nn.SampledSoftmaxLoss
import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.filePath
import org.apache.log4j.{Level, Logger}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DeepRetrievalTest extends AnyFlatSpec with Matchers {

  Logger.getLogger("com.mass").setLevel(Level.INFO)
  Engine.setCoreNumber(8)
  val numLayer = 3
  val numNode = 100
  val numPathPerItem = 2
  val trainBatchSize = 8192
  val evalBatchSize = 8192
  val seqLen = 10
  val minSeqLen = 2
  val dataPath = s"${filePath("deep-retrieval")}/data/data.csv"
  val mappingPath = s"${filePath("deep-retrieval")}/data/dr_mapping.txt"
  val initialize = true
  val splitRatio = 0.8
  val paddingIdx = -1
  val embedSize = 16
  val numEpoch = 2
  val reRankEpoch = Some(1)
  val learningRate = 7e-3
  val numSampled = 1
  val topk = 10
  val beamSize = 20
  val progressInterval = 50

  val dataset = LocalDataSet(
    numLayer = numLayer,
    numNode = numNode,
    numPathPerItem = numPathPerItem,
    trainBatchSize = trainBatchSize,
    evalBatchSize = evalBatchSize,
    seqLen = seqLen,
    minSeqLen = minSeqLen,
    dataPath = dataPath,
    mappingPath = mappingPath,
    initialize = true,
    splitRatio = splitRatio,
    delimiter = ","
  )

  val layerModel = LayerModel(dataset.numItem, numNode, numLayer, seqLen, embedSize)
  val reRankModel = RerankModel(dataset.numItem, seqLen, embedSize)
  val optimizer = LocalOptimizer(
    dataset = dataset,
    layerModel = layerModel,
    reRankModel = reRankModel,
    numEpoch = numEpoch,
    numLayer = numLayer,
    learningRate = learningRate,
    numSampled = numSampled,
    embedSize = embedSize,
    topk = topk,
    beamSize = beamSize,
    progressInterval = progressInterval,
    reRankEpoch = reRankEpoch
  )
  val optimResults = optimizer.optimize()

  val drModel = DeepRetrieval(
    layerModel = layerModel,
    reRankModel = reRankModel,
    numItem = dataset.numItem,
    numNode = numNode,
    numLayer = numLayer,
    seqLen = seqLen,
    embedSize = embedSize
  )
  val mappingOp = MappingOp(dataset.itemIdMapping, dataset.itemPathMapping)

  "DeepRetrieval model" should "initiate with correct classes" in {
    assert(getType(drModel) == "com.mass.dr.model.DeepRetrieval")
    layerModel shouldBe a [LayerModel]
    reRankModel shouldBe a [RerankModel]
    optimizer.layerCriterion shouldBe a [CrossEntropyLayer]
    optimizer.reRankCriterion shouldBe a [SampledSoftmaxLoss[_]]
  }

  "Loss in DeepRetrieval model" should "decrease during optimization" in {
    val evalResults = optimResults.map(_.meanMetrics)
    evalResults reduce { (former, later) =>
      former.layerLoss lazyZip later.layerLoss foreach((a, b) => assert(a >= b))
      assert(former.reRankLoss >= later.reRankLoss)
      later
    }
  }

  "DeepRetrieval model" should "return correct recommendations" in {
    val sequence = Seq(1, 2, 3, 4, 5, 6, 7, 89, 2628, 1681)
    val recommendResult = drModel.recommend(sequence, 10, 20, mappingOp).map(_._1)
    recommendResult.length should be >= 5
    val totalTime = (1 to 100).map(_ => time(drModel.recommend(sequence, 10, 20, mappingOp)))
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
