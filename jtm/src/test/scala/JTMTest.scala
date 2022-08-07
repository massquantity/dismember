import com.mass.jtm.optim.JTM
import com.mass.jtm.tree.TreeUtil
import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.filePath
import org.apache.log4j.{Level, Logger}
import org.scalatest.flatspec.AnyFlatSpec

class JTMTest extends AnyFlatSpec {
  Logger.getLogger("com.mass").setLevel(Level.INFO)

  val numThread = 8
  Engine.setCoreNumber(numThread)
  val prefix = s"${filePath("jtm")}/data/"
  val jtm = JTM(
    dataPath = prefix + "train_data.csv",
    treePath = prefix + "example_tree.bin",
    modelPath = prefix + "example_model.bin",
    gap = 2,
    seqLen = 10,
    hierarchical = false,
    minLevel = 0,
    numThreads = numThread,
    useMask = true
  )
  val projection = jtm.optimize()
  val treeMeta = TreeUtil.getTreeMeta(jtm)
  TreeUtil.writeTree(jtm, projection, prefix + "jtm_tree.bin")

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
}
