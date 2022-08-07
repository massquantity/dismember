import com.mass.dr.dataset.LocalDataSet
import com.mass.dr.model.{LayerModel, MappingOp}
import com.mass.dr.optim.CoordinateDescent
import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.filePath
import org.apache.log4j.{Level, Logger}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class CoordinateDescentTest extends AnyFlatSpec with Matchers {

  Logger.getLogger("com.mass").setLevel(Level.INFO)
  Engine.setCoreNumber(8)
  val numLayer = 3
  val numNode = 100
  val numPathPerItem = 2
  val batchSize = 8192
  val evalBatchSize = 8192
  val seqLen = 10
  val minSeqLen = 2
  val dataPath = s"${filePath("deep-retrieval")}/data/data.csv"
  val mappingPath = s"${filePath("deep-retrieval")}/data/dr_mapping.txt"
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

  val layerModel = LayerModel(dataset.numItem, numNode, numLayer, seqLen, embedSize)
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
    val itemPathMapping = coo.optimize(layerModel, trainMode = "batch")
    itemPathMapping.keySet should be (dataset.idItemMapping.keySet)
    all (itemPathMapping.values) should have length numPathPerItem
  }

  "CoordinateDescent streaming mode" should "update mapping correctly" in {
    val itemPathMapping = coo.optimize(layerModel, trainMode = "streaming")
    assert(itemPathMapping.keySet == dataset.idItemMapping.keySet)
    all (itemPathMapping.values) should have length numPathPerItem
  }
}
