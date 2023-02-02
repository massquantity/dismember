import java.io.File

import com.mass.otm.tree.TreeConstruction
import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.filePath
import com.mass.tdm.utils.Serialization
import org.apache.commons.io.FileUtils
import org.apache.log4j.{Level, Logger}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TreeConstructionSpec extends AnyFlatSpec with Matchers {

  Logger.getLogger("com.mass").setLevel(Level.INFO)
  Engine.setCoreNumber(8)

  val dataPath = s"${filePath("otm")}/data/example_data.csv"
  val modelPath = s"${filePath("otm")}/data/otm/example_model.bin"
  val mappingPath = s"${filePath("otm")}/data/otm/example_mapping.txt"
  val testPath = s"${filePath("otm")}/test_path"
  FileUtils.forceMkdir(new File(testPath))

  val tree = TreeConstruction(
    dataPath = dataPath,
    modelPath = modelPath,
    mappingPath = mappingPath,
    gap = 2,
    labelNum = 5,
    minSeqLen = 2,
    seqLen = 10,
    splitRatio = 0.8,
    numThreads = 8,
    useMask = true
  )
  val projection = tree.run()
  Serialization.saveMapping(s"$testPath/otm_mapping.txt", projection)

  "Final projection" should "have sufficient item ids" in {
    projection.size should === (tree.itemIdMapping.size)
    projection.keys.toSeq.sorted should equal (tree.itemIdMapping.keys.toSeq.sorted)
  }

  "Final projection" should "have correct range" in {
    val minLeafCode = math.pow(2, tree.leafLevel).toInt - 1
    val maxLeafCode = minLeafCode * 2
    projection.values.min should be >= minLeafCode
    projection.values.max should be <= maxLeafCode
  }

  FileUtils.deleteDirectory(FileUtils.getFile(testPath))
}
