package com.mass.sparkdl.nn

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.{AbstractModule, Activity}
import com.mass.sparkdl.nn.Graph.ModuleNode
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}
import com.mass.sparkdl.utils.{DirectedGraph, Node, T}

abstract class Graph[T: ClassTag](
    val inputs: Seq[ModuleNode[T]],
    val outputs: Seq[ModuleNode[T]],
    val variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None)(
    implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Activity, T] {

  protected val modules = new ArrayBuffer[AbstractModule[Activity, Activity, T]]()

  protected val dummyOutput: ModuleNode[T] = new ModuleNode[T](new Identity[T]())

  outputs.foreach(_ -> dummyOutput)

  protected val forwardGraph: DirectedGraph[AbstractModule[Activity, Activity, T]] =
    dummyOutput.graph(reverse = true)

  protected val forwardNodes: Array[Node[AbstractModule[Activity, Activity, T]]] =
    forwardGraph.DFS().toArray

  populateModules()

  protected var dummyOutputGrad: ModuleNode[T] = _
  protected var backwardGraph: DirectedGraph[AbstractModule[Activity, Activity, T]] = _
  protected var backwardNodes: Array[Node[AbstractModule[Activity, Activity, T]]] = _

  def populateModules(): Unit

  /*
  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    variables match {
      case None => super.parameters()
      case Some((weights, gradients)) => (weights, gradients)
    }
  }
  */

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    val weights = new ArrayBuffer[Tensor[T]]()
    val gradWeights = new ArrayBuffer[Tensor[T]]()
    modules.foreach(m => {
      val params = m.parameters()
      if (params != null) {
        params._1.foreach(weights += _)
        params._2.foreach(gradWeights += _)
      }
    })
    (weights.toArray, gradWeights.toArray)
  }

  def findFirstInput(node: ModuleNode[T], input: Activity): Activity = {
    if (inputs.length == 1) {
      require(inputs.head.eq(node), "input node is not in the input list")
      input
    } else {
      val i = inputs.indexOf(node)
      require(i != -1, "input node is not in the input list")
      input.toTable.apply[Tensor[T]](i)
    }
  }

  def findInput(curNode: ModuleNode[T]): Activity = {
    val preActivities = curNode.prevNodesAndEdges.map { case (node, edge) =>
      edge.fromIndex match {
        case Some(i) =>
          if (node.element.output == null || (i == 0 && node.element.output.isTensor)) {
            node.element.output
          } else {
            node.element.output.toTable.apply[Activity](i)
          }
        case None =>
          node.element.output
      }
    }

    if (preActivities.length == 1) {
      preActivities.head
    } else {
      T.seq(preActivities)
    }
  }

  protected def findGradOutput(curNode: ModuleNode[T], gradOutput: Activity): Activity = {
    var curGradOutput: Activity = if (curNode.eq(dummyOutputGrad)) gradOutput else null

    curNode.prevNodesAndEdges.foreach { case (node, edge) =>
      val otherActivity = {
        if (node.element.gradInput.isTensor || node.nextEdges.length == 1) {
          node.element.gradInput
        } else {
          val index = node.nextEdges.indexOf(edge)
          node.element.gradInput.toTable.apply[Activity](index)
        }
      }

      edge.fromIndex match {
        case Some(i) =>
          if (i == 0 && curNode.element.output.isTensor) {
            curGradOutput = accActivity(curGradOutput, otherActivity)
          } else {
            if (curNode.element.output.isTable && curGradOutput == null) {
              curGradOutput = T()
            }
            val curActivity = curGradOutput.toTable.getOrElse[Activity](i, null)
            curGradOutput.toTable(i) = accActivity(curActivity, otherActivity)
          }
        case None =>
          curGradOutput = accActivity(curGradOutput, otherActivity)
      }
    }
    curGradOutput
  }

  @inline
  protected def accActivity(activity: Activity, other: Activity): Activity = {
    if (activity == null) {
      other
    } else {
      if (other.isTensor) {
        require(activity.isTensor, "Cannot add a table to a tensor")
        activity.toTensor[T].add(other.toTensor[T])
      } else {
        val actTable = activity.toTable
        val otherTable = other.toTable
        otherTable.keySet.foreach(index => {
          if (actTable.contains(index)) {
            accActivity(actTable[Activity](index), otherTable[Activity](index))
          } else {
            actTable.insert(index.asInstanceOf[Int], otherTable(index))
          }
        })
        actTable
      }
    }
  }

  protected def fetchModelGradInput(): Activity = {
    if (inputs.length == 1) {
      inputs.head.element.gradInput
    } else {
      T.seq(inputs.map(node => node.element.gradInput))
    }
  }

  private[sparkdl] def buildBackwardGraph(): this.type = {
    val gradGraph = forwardGraph.cloneGraph(reverseEdge = true)
    dummyOutputGrad = gradGraph.source
    backwardNodes = gradGraph.DFS().filterNot(_.eq(dummyOutputGrad)).toArray

    val inputNames = inputs.map(_.element.getName).toSet
    val dummyBackwardEnd = Identity().inputs()
    val backwardTargets = backwardNodes.filter(n =>
      (n.element.parameters() != null && n.element.parameters()._1.length != 0)
        || inputNames.contains(n.element.getName)
    )
    backwardTargets.foreach(_ -> dummyBackwardEnd)
    backwardGraph = dummyBackwardEnd.graph(reverse = true)

    clearState()
    this
  }

  def node(name: String): ModuleNode[T] = {
    val matchNodes = forwardNodes.find(_.element.getName == name)
    matchNodes match {
      case Some(m) => m
      case None => throw new NoSuchElementException(s"Can not find node with name $name")
    }
  }

  override def training(): this.type = {
    train = true
    modules.foreach(_.training())
    this
  }

  override def getExtraParameter: Array[Tensor[T]] = {
    val extraParam = new ArrayBuffer[Tensor[T]]()
    modules.foreach(m => {
      val state = m.getExtraParameter
      if (state != null) {
        extraParam ++= state
      }
    })
    extraParam.toArray
  }

  override def clearState() : this.type = {
    super.clearState()
    modules.foreach(_.clearState())
    this
  }

  override def release(): Unit = {
    modules.foreach(_.release())
  }

  override def apply(name: String): Option[AbstractModule[Activity, Activity, T]] = {
    if (this.getName == name) {
      Some(this)
    } else {
      val found = modules.filter(m => m(name).isDefined)
      require(found.length <= 1, "find multiple modules with same name")
      found.headOption
    }
  }
}

object Graph {

  type ModuleNode[T] = Node[AbstractModule[Activity, Activity, T]]

  def apply[@specialized(Float, Double) T: ClassTag](
      input: Array[ModuleNode[T]],
      output: Array[ModuleNode[T]],
      variables: Option[(Array[Tensor[T]], Array[Tensor[T]])] = None)(
      implicit ev: TensorNumeric[T]): Graph[T] = {
    new StaticGraph[T](input, output, variables)
  }

  def apply[@specialized(Float, Double) T: ClassTag](
      input: ModuleNode[T],
      output: ModuleNode[T])(
      implicit ev: TensorNumeric[T]): Graph[T] = {
    new StaticGraph[T](Seq(input), Seq(output))
  }
}
