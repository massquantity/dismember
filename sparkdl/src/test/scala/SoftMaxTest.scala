import com.mass.sparkdl.nn.{LogSoftMax, SoftMax}
import com.mass.sparkdl.tensor.Tensor
import org.scalatest.flatspec.AnyFlatSpec

class SoftMaxTest extends AnyFlatSpec {

  val input: Tensor[Double] = Tensor[Double](Array(5.0, 2.0, 0.8, 0.3, 0.4, 1.0), Array(2, 3))
  val softmax: SoftMax[Double] = SoftMax[Double]()

  "SoftMax forward" should "output expected values as in PyTorch" in {
    val output = softmax.forward(input)
    val expectedOutput = Array(0.9392, 0.0468, 0.0141, 0.2428, 0.2683, 0.4889)
    assert(output.size() === input.size())
    output.storage().array().zip(expectedOutput) foreach { case (a, b) =>
      assert(math.abs(a - b) < 1e-4)
    }
  }

  "SoftMax backward" should "output expected values as in PyTorch" in {
    val gradOut = Tensor[Double](Array(0.5, 0.1, 0.6, 0.1, 0.9, 0.6), Array(2, 3))
    val gradInput = softmax.backward(input, gradOut)
    val expectedGradInput = Array(0.0162, -0.0179,  0.0017, -0.1115,  0.0915,  0.0200)
    assert(gradInput.size() === input.size())
    gradInput.storage().array().zip(expectedGradInput) foreach { case (a, b) =>
      assert(math.abs(a - b) < 1e-4)
    }
  }

  val input2: Tensor[Double] = Tensor[Double](Array(5.0, 2.0, 0.8, 0.3, 0.4, 1.0), Array(2, 3))
  val lsm: LogSoftMax[Double] = LogSoftMax[Double]()

  "LogSoftMax forward" should "output expected values as in PyTorch" in {
    val output = lsm.forward(input2)
    val expectedOutput = Array(-0.0628, -3.0628, -4.2628, -1.4156, -1.3156, -0.7156)
    assert(output.size() === input2.size())
    output.storage().array().zip(expectedOutput) foreach { case (a, b) =>
      assert(math.abs(a - b) < 1e-4)
    }

    val expectedSoftMaxOutput = Array(0.9392, 0.0468, 0.0141, 0.2428, 0.2683, 0.4889)
    output.storage().array().zip(expectedSoftMaxOutput) foreach { case (a, b) =>
      assert(math.abs(a - math.log(b)) < 1e-2)
    }
  }

  "LogSoftMax backward" should "output expected values as in PyTorch" in {
    val gradOut = Tensor[Double](Array(0.5, 0.1, 0.6, 0.1, 0.9, 0.6), Array(2, 3))
    val gradInput = lsm.backward(input2, gradOut)
    val expectedGradInput = Array(-0.6270,  0.0439,  0.5831, -0.2885,  0.4707, -0.1822)
    assert(gradInput.size() === input2.size())
    gradInput.storage().array().zip(expectedGradInput) foreach { case (a, b) =>
      assert(math.abs(a - b) < 1e-4)
    }
  }
}
