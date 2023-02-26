import java.io.File
import java.nio.file.{Files, Paths}

import scala.annotation.unused
import scala.reflect.runtime.universe.{typeTag, TypeTag}

import com.mass.dr.dataset.LocalDataSet
import com.mass.dr.loss.CrossEntropyLayer
import com.mass.dr.model.{DeepRetrieval, LayerModel, MappingOp, RerankModel}
import com.mass.dr.optim.LocalOptimizer
import com.mass.scalann.nn.SampledSoftmaxLoss
import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.filePath
import org.apache.commons.io.FileUtils
import org.apache.log4j.{Level, Logger}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DeepRetrievalSpec extends AnyFlatSpec with Matchers {

  Logger.getLogger("com.mass").setLevel(Level.INFO)
  Engine.setCoreNumber(8)

  val dataPath = s"${filePath("deep-retrieval")}/data/example_data.csv"
  val testPath = s"${filePath("deep-retrieval")}/test_path"
  FileUtils.forceMkdir(new File(testPath))

  val modelPath = s"$testPath/dr_model.bin"
  val mappingPath = s"$testPath/dr_mapping.bin"
  val numLayer = 3
  val numNode = 100
  val numPathPerItem = 2
  val trainBatchSize = 8192
  val evalBatchSize = 8192
  val seqLen = 10
  val minSeqLen = 2
  val initMapping = true
  val splitRatio = 0.8
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
    initMapping = initMapping,
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
    recommendResult.length should be >= 3
    val totalTime = (1 to 100).map(_ => time(drModel.recommend(sequence, 10, 20, mappingOp)))
    println(f"Average recommend time: ${totalTime.sum / totalTime.length}%.4fms")
  }

  "DeepRetrieval model" should "save and load correctly" in {
    val sequence = Seq(0, 0, 2126, 204, 3257, 3439, 996, 1681, 3438, 1882)
    val preRec = drModel.recommend(sequence, topk = 3, beamSize = 20, mappings = mappingOp).map(_._1)
    preRec should have length 3

    MappingOp.writeMapping(mappingPath, dataset.itemIdMapping, dataset.itemPathMapping)
    DeepRetrieval.saveModel(drModel, modelPath)
    assert(Files.exists(Paths.get(mappingPath)))
    assert(Files.exists(Paths.get(modelPath)))

    val loadedMappingOP = MappingOp.loadMappingOp(mappingPath)
    val loadedModel = DeepRetrieval.loadModel(modelPath)
    val newRec = loadedModel.recommend(sequence, topk = 3, beamSize = 20, mappings = loadedMappingOP).map(_._1)
    newRec should have length 3
    preRec should equal(newRec)

    FileUtils.deleteDirectory(FileUtils.getFile(testPath))
  }

  def getType[T: TypeTag](@unused obj: T): String = typeTag[T].tpe.toString

  def time[T](block: => T): Double = {
    val start = System.nanoTime()
    val _ = block
    val end = System.nanoTime()
    (end - start) / 1e6d
  }
}
