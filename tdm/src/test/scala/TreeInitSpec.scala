import java.io.File
import java.nio.file.{Files, Paths}

import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.filePath
import com.mass.tdm.tree.TreeInit
import org.apache.commons.io.FileUtils
import org.apache.log4j.{Level, Logger}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TreeInitSpec extends AnyFlatSpec with Matchers {

  Logger.getLogger("com.mass").setLevel(Level.INFO)
  Engine.setCoreNumber(8)

  val dataPath = s"${filePath("tdm")}/data/example_data.csv"
  val testPath = s"${filePath("tdm")}/test_path"
  FileUtils.forceMkdir(new File(testPath))

  "Tree initialization" should "construct tree correctly" in {
    val trainPath = s"$testPath/train_data.csv"
    val evalPath = s"$testPath/eval_data.csv"
    val statPath = s"$testPath/stat_data.txt"
    val leafIdPath = s"$testPath/leaf_id_data.txt"
    val treePath = s"$testPath/tdm_tree.bin"
    val userConsumedPath = s"$testPath/user_consumed.txt"

    val tree = new TreeInit(
      seqLen = 10,
      minSeqLen = 2,
      splitForEval = true,
      splitRatio = 0.8
    )
    val (ids, codes) = tree.generate(
      dataFile = dataPath,
      trainFile = trainPath,
      evalFile = Some(evalPath),
      statFile = statPath,
      leafIdFile = leafIdPath,
      treePbFile = treePath,
      userConsumedFile = Some(userConsumedPath)
    )
    val minCode = getMinCode(codes)

    ids.length shouldEqual codes.length
    all(codes.toSeq) should be >= minCode
    assert(Files.exists(Paths.get(treePath)))

    FileUtils.deleteDirectory(FileUtils.getFile(testPath))
  }

  def getMinCode(treeCodes: Array[Int]): Int = {
    val log2 = (n: Int) => math.floor(math.log(n) / math.log(2)).toInt
    val maxLevel = log2(treeCodes.max + 1)
    math.pow(2, maxLevel - 1).toInt - 1
  }
}
