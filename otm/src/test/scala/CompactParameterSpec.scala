import com.mass.otm.model.{DIN, ModelUtil}
import org.scalatest.flatspec.AnyFlatSpec

class CompactParameterSpec extends AnyFlatSpec {

  val model = DIN.buildModel[Double](4, 100)

  "Weights in model" should "be compact and share same underlying storage" in {
    ModelUtil.compactParameters(model)
    val (weightParams, _) = model.parameters()
    val weightSizes = ModelUtil.cumSum(weightParams.toSeq.map(_.nElement()))
    val offset = weightParams.head.storageOffset()
    weightParams.tail zip weightSizes.init foreach { case (w, len) =>
      assert(weightParams.head.storage() eq w.storage())
      assert((offset + len) == w.storageOffset())
    }
    val (compactWeights, _) = ModelUtil.getParameters(model)
    assert(compactWeights.nElement() == weightParams.map(_.nElement()).sum)
  }

  "Parameters in model" should "be cleared" in {
    ModelUtil.clearParameters(model)
    val (weightParams, gradParams) = model.parameters()
    weightParams.foreach(w => assert(w.isEmpty))
    gradParams.foreach(g => assert(g.isEmpty))
  }
}
