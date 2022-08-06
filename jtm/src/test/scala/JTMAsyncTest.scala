import com.mass.jtm.optim.JTMAsync
import com.mass.jtm.tree.TreeUtil
import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.filePath
import org.apache.log4j.{Level, Logger}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.Inspectors.forAll

class JTMAsyncTest extends AnyFlatSpec with Matchers {
  Logger.getLogger("com.mass").setLevel(Level.INFO)

  val numThread = 8
  Engine.setCoreNumber(numThread)
  val prefix = s"${filePath("jtm")}/data/"
  val jtm = JTMAsync(
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
}
