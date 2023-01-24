import java.io.File
import java.nio.file.{Files, Paths}

import com.mass.scalann.nn.BCECriterionWithLogits
import com.mass.scalann.optim.Adam
import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.filePath
import com.mass.tdm.dataset.LocalDataSet
import com.mass.tdm.model.{DIN, TDM}
import com.mass.tdm.optim.LocalOptimizer
import com.mass.tdm.tree.TreeInit
import org.apache.commons.io.FileUtils
import org.apache.log4j.{Level, Logger}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TdmModelTrainSpec extends AnyFlatSpec with Matchers {

  Logger.getLogger("com.mass").setLevel(Level.INFO)
  Engine.setCoreNumber(8)

  val dataPath = s"${filePath("tdm")}/data/example_data.csv"
  val testPath = s"${filePath("tdm")}/test_path"
  FileUtils.forceMkdir(new File(testPath))

  val trainPath = s"$testPath/train_data.csv"
  val evalPath = s"$testPath/eval_data.csv"
  val statPath = s"$testPath/stat_data.txt"
  val leafIdPath = s"$testPath/leaf_id_data.txt"
  val treePath = s"$testPath/tdm_tree.bin"
  val userConsumedPath = s"$testPath/user_consumed.txt"
  val modelPath = s"$testPath/tdm_model.bin"
  val embedPath = s"$testPath/embed.csv"
  val trainBatchSize = 8192
  val evalBatchSize = 8192
  val layerNegCounts = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,19,22,25,30,76,200"
  val embedSize = 16
  val learningRate = 7e-3
  val numIteration = 100
  val progressInterval = 50

  // initialize tree first
  new TreeInit(
    seqLen = 10,
    minSeqLen = 2,
    splitForEval = true,
    splitRatio = 0.8
  ).generate(
    dataFile = dataPath,
    trainFile = trainPath,
    evalFile = Some(evalPath),
    statFile = statPath,
    leafIdFile = leafIdPath,
    treePbFile = treePath,
    userConsumedFile = Some(userConsumedPath)
  )

  val dataset = LocalDataSet(
    trainPath = trainPath,
    evalPath = evalPath,
    pbFilePath = treePath,
    userConsumedPath = userConsumedPath,
    totalTrainBatchSize = trainBatchSize,
    totalEvalBatchSize = evalBatchSize,
    seqLen = 10,
    layerNegCounts = layerNegCounts,
    withProb = false,
    numThreads = 8,
    useMask = true
  )
  val deepModel = DIN.buildModel[Float](embedSize)
  LocalOptimizer(
    model = deepModel,
    dataset = dataset,
    criterion = BCECriterionWithLogits(),
    optimMethod = Adam(learningRate = learningRate),
    numIteration = numIteration,
    progressInterval = progressInterval,
    topk = 10,
    candidateNum = 20,
    useMask = true
  ).optimize()

  val tdmModel = TDM(deepModel, "din")
  val sequence = Array(0, 0, 2126, 204, 3257, 3439, 996, 1681, 3438, 1882)
  val preRec = tdmModel.recommend(sequence, topk = 3, candidateNum = 20).map(_._1)
  preRec should have length 3

  TDM.saveModel(modelPath, embedPath, deepModel, embedSize)
  assert(Files.exists(Paths.get(modelPath)))

  TDM.loadTree(treePath)
  val loadedModel = TDM.loadModel(modelPath, "din")
  val newRec = loadedModel.recommend(sequence, topk = 3, candidateNum = 20).map(_._1)
  newRec should have length 3
  preRec should equal(newRec)

  FileUtils.deleteDirectory(FileUtils.getFile(testPath))
}
