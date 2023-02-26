import java.io.File
import java.nio.file.{Files, Paths}

import com.mass.dr.dataset.LocalDataSet
import com.mass.dr.model.{DeepRetrieval, MappingOp}
import com.mass.dr.optim.CoordinateDescent
import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.filePath
import org.apache.commons.io.FileUtils
import org.apache.log4j.{Level, Logger}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CoordinateDescentSpec extends AnyFlatSpec with Matchers {

  Logger.getLogger("com.mass").setLevel(Level.INFO)
  Engine.setCoreNumber(8)

  val dataPath = s"${filePath("deep-retrieval")}/data/example_data.csv"
  val modelPath = s"${filePath("deep-retrieval")}/data/dr/example_model.bin"
  val testPath = s"${filePath("deep-retrieval")}/test_path"
  FileUtils.forceMkdir(new File(testPath))

  val mappingPath = s"$testPath/dr_mapping.bin"
  val numLayer = 3
  val numNode = 100
  val numPathPerItem = 2
  val batchSize = 8192
  val evalBatchSize = 8192
  val seqLen = 10
  val minSeqLen = 2
  val initialize = true
  val splitRatio = 0.8
  val embedSize = 16
  val numCandidatePath = 20
  val numIteration = 3
  val decayFactor = 0.999
  val penaltyFactor = 3e-6
  val penaltyPolyOrder = 4

  val dataset = LocalDataSet(
    numLayer = numLayer,
    numNode = numNode,
    numPathPerItem = numPathPerItem,
    trainBatchSize = batchSize,
    evalBatchSize = evalBatchSize,
    seqLen = seqLen,
    minSeqLen = minSeqLen,
    dataPath = dataPath,
    mappingPath = mappingPath,
    initMapping = initialize,
    splitRatio = splitRatio,
    delimiter = ","
  )

  val drModel = DeepRetrieval.loadModel(modelPath)
  val coo = CoordinateDescent(
    dataset = dataset,
    batchSize = batchSize,
    numIteration = numIteration,
    numCandidatePath = numCandidatePath,
    numPathPerItem = numPathPerItem,
    numLayer = numLayer,
    numNode = numNode,
    decayFactor = decayFactor,
    penaltyFactor = penaltyFactor,
    penaltyPolyOrder = penaltyPolyOrder
  )

  "CoordinateDescent batch mode" should "update mapping correctly" in {
    val itemPathMapping = coo.optimize(drModel.layerModel, trainMode = "batch")
    itemPathMapping.keySet should be (dataset.idItemMapping.keySet)
    all (itemPathMapping.values) should have length numPathPerItem
  }

  "CoordinateDescent streaming mode" should "update mapping correctly" in {
    val itemPathMapping = coo.optimize(drModel.layerModel, trainMode = "streaming")
    assert(itemPathMapping.keySet == dataset.idItemMapping.keySet)
    all (itemPathMapping.values) should have length numPathPerItem

    MappingOp.writeMapping(mappingPath, dataset.itemIdMapping, itemPathMapping)
    assert(Files.exists(Paths.get(mappingPath)))

    val sequence = Seq(0, 0, 2126, 204, 3257, 3439, 996, 1681, 3438, 1882)
    val loadedMappingOP = MappingOp.loadMappingOp(mappingPath)
    val newRec = drModel.recommend(sequence, topk = 3, beamSize = 20, mappings = loadedMappingOP).map(_._1)
    newRec should have length 3
    FileUtils.deleteDirectory(FileUtils.getFile(testPath))
  }
}
