package com.mass.sparkdl.nn

import scala.reflect.ClassTag

import com.mass.sparkdl.nn.Graph.ModuleNode
import com.mass.sparkdl.nn.abstractnn.{AbstractModule, Activity}
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}
import com.mass.sparkdl.utils.Node

class StaticGraph[T: ClassTag](
    private val _inputs: Seq[ModuleNode[T]],
    private val _outputs: Seq[ModuleNode[T]],
    private val _variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None)(
    implicit ev: TensorNumeric[T]) extends Graph[T](_inputs, _outputs, _variables) {

  private val forwardExecution = forwardGraph.topologicalSort().reverse
  private var backwardExecution: Array[Node[AbstractModule[Activity, Activity, T]]] = _
  private val inputCache = new Array[Activity](forwardExecution.length)
  private var backId2ForwardId: Array[Int] = _
  private var gradOutputCache: Array[Activity] = _

  buildBackwardGraph()

  override def updateOutput(input: Activity): Activity = {
    var i = 0
    while (i < forwardExecution.length) {
      val node = forwardExecution(i)
      val nodeInput = {
        if (node.prevNodes.isEmpty) {
          findFirstInput(node, input)
        } else {
          findInput(node)
        }
      }

      inputCache(i) = nodeInput
      node.element.forward(nodeInput)
      i += 1
    }
    output = dummyOutput.element.output
    output
  }

  override def backward(input: Activity, gradOutput: Activity): Activity = {
    val gradients = backwardImpl(input, gradOutput, executeBackward = true)
    gradients
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    backwardImpl(input, gradOutput, executeBackward = false)
  }

  override def buildBackwardGraph(): this.type = {
    super.buildBackwardGraph()
    backwardExecution = backwardGraph.topologicalSort().reverse
    backId2ForwardId = new Array[Int](backwardExecution.length)
    gradOutputCache = new Array[Activity](backwardExecution.length)

    var i = 0
    // exclude dummy output
    while (i < backwardExecution.length - 1) {
      var j = 0
      var found = false
      while (j < forwardExecution.length) {
        if (forwardExecution(j).element.getName == backwardExecution(i).element.getName) {
          backId2ForwardId(i) = j
          found = true
        }
        j += 1
      }
      require(found, "could not find backward layer in forward executions.")
      i += 1
    }
    this
  }

  override def accGradParameters(input: Activity, gradOutput: Activity): Unit = {
    var i = 0
    while (i < backwardExecution.length - 1) {
      val curNode = backwardExecution(i)
      val curInput = inputCache(backId2ForwardId(i))
      curNode.element.accGradParameters(curInput, gradOutputCache(i))
      i += 1
    }
  }

  override def populateModules(): Unit = {
    modules ++= {
      forwardGraph
        .topologicalSort()
        .filter(n => !n.eq(dummyOutput))
        .map(_.element)
        .reverse
    }
  }

  private def backwardImpl(
      input: Activity,
      gradOutput: Activity,
      executeBackward: Boolean): Activity = {

    dummyOutputGrad.element.gradInput = gradOutput
    var i = 0
    while (i < backwardExecution.length - 1) {
      val curNode = backwardExecution(i)
      val curGradOutput = findGradOutput(curNode, gradOutput)
      gradOutputCache(i) = curGradOutput
      val curInput = inputCache(backId2ForwardId(i))
      if (executeBackward) {
        curNode.element.backward(curInput, curGradOutput)
      } else {
        curNode.element.updateGradInput(curInput, curGradOutput)
      }
      i += 1
    }

    gradInput = fetchModelGradInput()
    gradInput
  }
}
