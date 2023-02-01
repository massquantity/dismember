import java.io.File

import com.mass.jtm.optim.JTM
import com.mass.jtm.tree.TreeUtil
import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.filePath
import org.apache.commons.io.FileUtils
import org.apache.log4j.{Level, Logger}
import org.scalatest.flatspec.AnyFlatSpec

class JtmSpec extends AnyFlatSpec {

  Logger.getLogger("com.mass").setLevel(Level.INFO)
  Engine.setCoreNumber(8)

  val dataPath = s"${filePath("jtm")}/data/jtm/train_data.csv"
  val treePath = s"${filePath("jtm")}/data/jtm/example_tree.bin"
  val modelPath = s"${filePath("jtm")}/data/jtm/example_model.bin"
  val testPath = s"${filePath("jtm")}/test_path"
  FileUtils.forceMkdir(new File(testPath))

  val jtm = JTM(
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
    assert(projection.size == treeMeta.leafNum)
  }

  "Final projection" should "have sufficient item ids" in {
    assert(projection.size == treeMeta.itemIds.length)
    assert(projection.keys.forall(treeMeta.itemIds.contains))
  }

  "Final projection" should "have correct range" in {
    val minLeafCode = math.pow(2, treeMeta.maxLevel).toInt - 1
    val maxLeafCode = minLeafCode * 2
    assert(projection.values.min >= minLeafCode)
    assert(projection.values.max <= maxLeafCode)
  }

  FileUtils.deleteDirectory(FileUtils.getFile(testPath))
}
