import com.mass.scalann.nn.SampledSoftmaxLoss
import com.mass.scalann.tensor.Tensor
import org.scalatest.flatspec.AnyFlatSpec

class SampledSoftmaxLossTest extends AnyFlatSpec {

  val batchSize = 6
  val embedSize = 10
  val numSampled = 4
  val numClasses = 200
  val learningRate = 7e-3
  val inputVecs: Tensor[Double] = Tensor[Double](batchSize, embedSize).rand(-0.05, 0.05, seed = 2022)
  val weights: Tensor[Double] = Tensor[Double](numClasses, embedSize).randn(0.0, 0.01, seed = 2022)
  val biases: Tensor[Double] = Tensor[Double](numClasses).zero()
  val positiveItems: Array[Int] = Array(0, 1, 3, 2, 77, 101)
  val negativeItems: Array[List[Int]] = Array(
    List(19, 3, 66, 190),
    List(33, 4, 88, 111),
    List(2, 48, 92, 129),
    List(1, 66, 34, 167),
    List(53, 11, 0, 123),
    List(8, 99, 100, 12)
  )
  // positive + negative items
  val sampledItems: Array[Int] = positiveItems zip negativeItems flatMap { case (pos, negs) => pos :: negs }
  val targetTensor: Tensor[Double] = Tensor[Int](positiveItems, Array(batchSize)).asInstanceOf[Tensor[Double]]
  val sampledSoftmax: SampledSoftmaxLoss[Double] = SampledSoftmaxLoss[Double](
    numSampled = numSampled,
    numClasses = numClasses,
    embedSize = embedSize,
    learningRate = learningRate,
    weights = weights,
    biases = biases,
    batchMode = false,
    sampledValues = Some(sampledItems)
  )

  "Loss in SampledSoftmax" should "decrease during optimization" in {
    val losses = 1 to 7 map { _ =>
      val loss = sampledSoftmax.forward(inputVecs, targetTensor)
      val gradInput = sampledSoftmax.backward(inputVecs, Tensor[Double](batchSize).zero())
      assert(gradInput.size() === inputVecs.size())
      loss
    }
    losses reduce { (formerLoss, laterLoss) =>
      assert(formerLoss > laterLoss)
      laterLoss
    }
  }
}
