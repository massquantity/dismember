import com.mass.otm.model.{DIN, ModelUtil}
import org.scalatest.flatspec.AnyFlatSpec

class CloneModelSpec extends AnyFlatSpec {

  val model = DIN.buildModel[Double](4, 100)
  ModelUtil.compactParameters(model)

  val modelWeights = ModelUtil.extractWeights(model)
  ModelUtil.clearParameters(model)
  val clonedModels = (1 to 8).map { _ =>
    val m = model.cloneModule()
    ModelUtil.putWeights(m, modelWeights)
    ModelUtil.initGradients(m, modelWeights)
    m
  }
  ModelUtil.putWeights(model, modelWeights)
  ModelUtil.initGradients(model, modelWeights)

  "Cloned models" should "share same weights storage" in {
    val (weights, _) = model.parameters()
    for {
      cm <- clonedModels
      (clonedWeights, _) = cm.parameters()
      (a, b) <- clonedWeights zip weights
    } assert(a.storage() === b.storage())
  }

  "Cloned models" should "have different gradients storage" in {
    val (_, gradients) = model.parameters()
    for {
      cm <- clonedModels
      (_, clonedGradients) = cm.parameters()
      (a, b) <- clonedGradients zip gradients
    } assert(a.storage() !== b.storage())
  }
}
