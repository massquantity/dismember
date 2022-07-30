import com.mass.scalann.nn.{ClassNLLCriterion, CrossEntropyCriterion}
import com.mass.scalann.tensor.Tensor
import org.scalatest.flatspec.AnyFlatSpec

class CrossEntropyTest extends AnyFlatSpec {

  val input: Tensor[Double] = Tensor[Double](Array(5.0, 2.0, 0.8, 0.3, 0.4, 1.0), Array(2, 3))
  val target: Tensor[Double] = Tensor[Double](Array(0.0, 1.0), Array(2))
  val classNLL: ClassNLLCriterion[Double] = ClassNLLCriterion()

  "ClassNLLCriterion forward" should "output expected values as in PyTorch" in {
    val outputLoss = classNLL.forward(input, target)
    val expectedLoss = -2.7
    assert(math.abs(outputLoss - expectedLoss) < 1e-4)
  }

  "ClassNLLCriterion backward" should "output expected values as in PyTorch" in {
    val gradInput = classNLL.backward(input, target)
    val expectedGradInput = Array(-0.5, 0.0, 0.0, 0.0, -0.5, 0.0)
    assert(gradInput.size() === input.size())
    gradInput.storage().array().zip(expectedGradInput) foreach { case (a, b) =>
      assert(math.abs(a - b) < 1e-4)
    }
  }

  val input2: Tensor[Double] = Tensor[Double](Array(2.3, -10.2, 0.8, -3.1, -2.2, 1.0), Array(3, 2))
  val target2: Tensor[Double] = Tensor[Double](Array(0.0, 1.0, 1.0), Array(3, 1))
  val crossEntropy: CrossEntropyCriterion[Double] = CrossEntropyCriterion()

  "CrossEntropyCriterion forward" should "output expected values as in PyTorch" in {
    val outputLoss = crossEntropy.forward(input2, target2)
    val expectedLoss = 1.3200
    assert(math.abs(outputLoss - expectedLoss) < 1e-4)
  }

  "CrossEntropyCriterion backward" should "output expected values as in PyTorch" in {
    val gradInput = crossEntropy.backward(input2, target2)
    val expectedGradInput = Array(-1.2219e-06,  1.2422e-06, 3.2672e-01, -3.2672e-01, 1.3055e-02, -1.3055e-02)
    assert(gradInput.size() === input2.size())
    gradInput.storage().array().zip(expectedGradInput) foreach { case (a, b) =>
      assert(math.abs(a - b) < 1e-4)
    }
  }
}
