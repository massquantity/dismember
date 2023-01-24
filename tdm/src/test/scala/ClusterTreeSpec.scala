import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.filePath
import com.mass.tdm.cluster.RecursiveCluster
import com.mass.tdm.utils.Utils.time
import org.apache.log4j.{Level, Logger}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ClusterTreeSpec extends AnyFlatSpec with Matchers {

  Logger.getLogger("com.mass").setLevel(Level.INFO)
  Engine.setCoreNumber(8)
  val treePath = s"${filePath("tdm")}/data/cluster_tree_path.txt"

  "Tree kmeans clustering" should "construct tree correctly" in {
    val model = RecursiveCluster(
      numItem = 5000,
      embedSize = 16,
      parallel = true,
      numThreads = Engine.coreNumber(),
      clusterIterNum = 10,
      clusterType = "kmeans"
    )
    val (ids, codes) = time(model.run(treePath), "tree kmeans clustering")
    val minCode = getMinCode(codes)
    ids.length shouldEqual codes.length
    all(codes.toSeq) should be >= minCode
  }

  "Tree spectral clustering" should "construct tree correctly" in {
    val model = RecursiveCluster(
      numItem = 5000,
      embedSize = 16,
      parallel = false,
      numThreads = Engine.coreNumber(),
      clusterIterNum = 10,
      clusterType = "spectral"
    )
    val (ids, codes) = time(model.run(treePath), "tree spectral clustering")
    val minCode = getMinCode(codes)
    ids.length shouldEqual codes.length
    all(codes.toSeq) should be >= minCode
  }

  def getMinCode(treeCodes: Array[Int]): Int = {
    val log2 = (n: Int) => math.floor(math.log(n) / math.log(2)).toInt
    val maxLevel = log2(treeCodes.max + 1)
    math.pow(2, maxLevel - 1).toInt - 1
  }
}
