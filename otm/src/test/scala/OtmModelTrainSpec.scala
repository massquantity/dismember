import java.io.File
import java.nio.file.{Files, Paths}

import com.mass.otm.dataset.LocalDataSet
import com.mass.otm.model.{DIN, OTM}
import com.mass.otm.optim.LocalOptimizer
import com.mass.otm.DeepModel
import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.filePath
import org.apache.commons.io.FileUtils
import org.apache.log4j.{Level, Logger}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class OtmModelTrainSpec extends AnyFlatSpec with Matchers {

  Logger.getLogger("com.mass").setLevel(Level.INFO)
  Engine.setCoreNumber(8)

  val dataPath = s"${filePath("otm")}/data/example_data.csv"
  val testPath = s"${filePath("otm")}/test_path"
  val modelPath = s"$testPath/otm_model.bin"
  val mappingPath = s"$testPath/otm_mapping.txt"
  val seqLen = 10
  val minSeqLen = 2
  val splitRatio = 0.8
  val leafInitMode = "random"
  val initMapping = true
  val labelNum = 5
  val dataMode = "default"
  // val targetMode = "pseudo"
  val seed = 2022
  val embedSize = 4
  val numEpoch = 1
  val trainBatchSize = 8192
  val evalBatchSize = 8192
  val learningRate = 7e-3
  val topk = 10
  val beamSize = 20
  val useMask = true
  val progressInterval = 200

  "OTM model with pseudo target" should "output correct recommendation" in {
    FileUtils.forceMkdir(new File(testPath))
    val targetMode = "pseudo"
    val (otmModel, deepModel, dataset) = trainModel(targetMode)
    val sequence = Seq(0, 0, 2126, 204, 3257, 3439, 996, 1681, 3438, 1882)
    val preRec = otmModel.recommend(sequence, topk = 3, beamSize = 20).map(_._1)
    preRec should have length 3

    OTM.saveModel(modelPath, mappingPath, deepModel, dataset.itemIdMapping)
    assert(Files.exists(Paths.get(modelPath)))
    assert(Files.exists(Paths.get(mappingPath)))

    val loadedModel = OTM.loadModel(modelPath, mappingPath, "DIN")
    val newRec = loadedModel.recommend(sequence, topk = 3, beamSize = 20).map(_._1)
    newRec should have length 3
    preRec should equal(newRec)
    FileUtils.deleteDirectory(FileUtils.getFile(testPath))
  }

  "OTM model with normal target" should "output correct recommendation" in {
    FileUtils.forceMkdir(new File(testPath))
    val targetMode = "normal"
    val (otmModel, deepModel, dataset) = trainModel(targetMode)
    val sequence = Seq(0, 0, 2126, 204, 3257, 3439, 996, 1681, 3438, 1882)
    val preRec = otmModel.recommend(sequence, topk = 3, beamSize = 20).map(_._1)
    preRec should have length 3

    OTM.saveModel(modelPath, mappingPath, deepModel, dataset.itemIdMapping)
    assert(Files.exists(Paths.get(modelPath)))
    assert(Files.exists(Paths.get(mappingPath)))

    val loadedModel = OTM.loadModel(modelPath, mappingPath, "DIN")
    val newRec = loadedModel.recommend(sequence, topk = 3, beamSize = 20).map(_._1)
    newRec should have length 3
    preRec should equal(newRec)
    FileUtils.deleteDirectory(FileUtils.getFile(testPath))
  }

  def trainModel(targetType: String): (OTM, DeepModel[Double], LocalDataSet) = {
    val dataset = LocalDataSet(
      dataPath = dataPath,
      seqLen = seqLen,
      minSeqLen = minSeqLen,
      splitRatio = splitRatio,
      leafInitMode = leafInitMode,
      initMapping = initMapping,
      mappingPath = mappingPath,
      labelNum = labelNum,
      seed = seed,
      dataMode = dataMode
    )
    val deepModel = DIN.buildModel[Double](embedSize, dataset.numTreeNode)
    LocalOptimizer(
      deepModel = deepModel,
      dataset = dataset,
      targetMode = targetType,
      numEpoch = numEpoch,
      totalTrainBatchSize = trainBatchSize,
      totalEvalBatchSize = evalBatchSize,
      learningRate = learningRate,
      beamSize = beamSize,
      topk = topk,
      seqLen = seqLen,
      useMask = useMask,
      progressInterval = progressInterval
    ).optimize()

    val otmModel = OTM(deepModel, dataset.itemIdMapping, "DIN")
    (otmModel, deepModel, dataset)
  }
}
