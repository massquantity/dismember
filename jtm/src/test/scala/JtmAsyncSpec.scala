import java.io.File

import com.mass.jtm.optim.JTMAsync
import com.mass.jtm.tree.TreeUtil
import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.filePath
import org.apache.commons.io.FileUtils
import org.apache.log4j.{Level, Logger}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Inspectors.forAll

class JtmAsyncSpec extends AnyFlatSpec with Matchers {

  Logger.getLogger("com.mass").setLevel(Level.INFO)
  Engine.setCoreNumber(8)

  val dataPath = s"${filePath("jtm")}/data/jtm/train_data.csv"
  val treePath = s"${filePath("jtm")}/data/jtm/example_tree.bin"
  val modelPath = s"${filePath("jtm")}/data/jtm/example_model.bin"
  val testPath = s"${filePath("jtm")}/test_path"
  FileUtils.forceMkdir(new File(testPath))

  val jtm = JTMAsync(
    dataPath = dataPath,
    treePath = treePath,
    modelPath = modelPath,
    gap = 2,
    seqLen = 10,
    hierarchical = false,
    minLevel = 0,
    numThreads = 8,
    useMask = true
  )
  val projection = jtm.optimize()
  val treeMeta = TreeUtil.getTreeMeta(jtm)
  TreeUtil.writeTree(jtm, projection, s"$testPath/jtm_tree.bin")

  "Final projection" should "have correct leaf size" in {
    projection should have size treeMeta.leafNum
  }

  "Final projection" should "have sufficient item ids" in {
    projection.size should === (treeMeta.itemIds.length)
    forAll(projection.keys)(treeMeta.itemIds.contains)
  }

  "Final projection" should "have correct range" in {
    val minLeafCode = math.pow(2, treeMeta.maxLevel).toInt - 1
    val maxLeafCode = minLeafCode * 2
    projection.values.min should be >= minLeafCode
    projection.values.max should be <= maxLeafCode
  }

  FileUtils.deleteDirectory(FileUtils.getFile(testPath))
}
