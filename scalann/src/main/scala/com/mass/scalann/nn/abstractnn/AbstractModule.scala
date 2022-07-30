package com.mass.scalann.nn.abstractnn

import scala.reflect.ClassTag

import com.mass.scalann.nn.graphnn.Graph.ModuleNode
import com.mass.scalann.nn.graphnn.Edge
import com.mass.scalann.nn.mixin.Module
import com.mass.scalann.optim.OptimMethod
import com.mass.scalann.tensor.{Tensor, TensorDataType, TensorNumeric}
import org.apache.commons.lang3.SerializationUtils

abstract class AbstractModule[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag](
    implicit ev: TensorNumeric[T]) extends Serializable {

  var output: B = Activity.allocate[B, T]()
  var gradInput: A = Activity.allocate[A, T]()

  final def forward(input: A): B = {
    try {
      updateOutput(input)
    } catch {
      case t: Throwable =>
        throw t
    }
    output
  }

  def backward(input: A, gradOutput: B): A = {
    updateGradInput(input, gradOutput)
    accGradParameters(input, gradOutput)
    gradInput
  }

  def updateOutput(input: A): B

  def updateGradInput(input: A, gradOutput: B): A

  def accGradParameters(input: A, gradOutput: B): Unit = { }

  def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = null

  def zeroGradParameters(): Unit = {
    val params = parameters()
    if (params != null) {
      params._1 zip params._2 foreach { case (weight, grad) =>
        grad.resizeAs(weight).zero()
      }
    }
  }

  def inputs(nodes: ModuleNode[T]*): ModuleNode[T] = {
    processInputs(nodes)
  }

  def inputs(nodesWithIndex: Seq[(ModuleNode[T], Int)], nodes: ModuleNode[T]*): ModuleNode[T] = {
    val curNode = new ModuleNode[T](this)
    if (nodesWithIndex != null) {
      nodesWithIndex.foreach { case (node, index) =>
        node.add(curNode, Edge(index))
      }
    }
    if (nodes != null) {
      nodes.foreach(node => node.add(curNode, Edge()))
    }
    curNode
  }

  def inputs(node: ModuleNode[T], preNode: ModuleNode[T], name: String): ModuleNode[T] = {
    preNode.find(name) match {
      case Some(n) =>
        node.add(n, Edge())
      case None =>
        val curNode = new ModuleNode[T](this)
        curNode.setName(name)
        node.add(curNode, Edge())
    }
  }

  protected def processInputs(nodes: Seq[ModuleNode[T]]): ModuleNode[T] = {
    val curNode = new ModuleNode[T](this)
    nodes.foreach(node => node.add(curNode, Edge()))
    curNode
  }

  final def getName: String = {
    if (this.name == null) {
      s"${this.getClass.getSimpleName}$namePostfix"
    } else {
      this.name
    }
  }

  final def setName(name : String) : this.type = {
    this.name = name
    this
  }

  def apply(name: String): Option[AbstractModule[A, B, T]] = {
    if (this.getName == name) Some(this) else None
  }

  def clearState(): this.type = {
    output match {
      case _: Tensor[_] =>
        output.asInstanceOf[Tensor[_]].set()
      case _ =>
    }
    gradInput match {
      case _: Tensor[_] =>
        gradInput.asInstanceOf[Tensor[_]].set()
      case _ =>
    }
    this
  }

  def training(): this.type = {
    train = true
    this
  }

  def evaluate(): this.type = {
    train = false
    this
  }

  final def isTraining: Boolean = {
    this.train
  }

  final def cloneModule(): this.type = {
    SerializationUtils.clone(this)
  }

  final def getNumericType: TensorDataType = {
    ev.getType
  }

  def release(): Unit = { }

  private val namePostfix = Integer.toHexString(java.util.UUID.randomUUID().hashCode())

  protected var scaleW: Double = 1.0
  protected var scaleB: Double = 1.0

  private var name: String = _
  private var id: Int = 0

  protected var train: Boolean = true

  private var _optimMethod: OptimMethod[T] = _

  private[scalann] def setOptimMethod(optimMethod: OptimMethod[T]): Unit = {
    _optimMethod = optimMethod
  }

  private[scalann] def getOptimMethod: OptimMethod[T] = _optimMethod

  def setId(id: Int): Unit = {
    this.id = id
  }

  final def adjustParameters(): (Tensor[T], Tensor[T]) = {
    val (weightParameters, gradParameters) = this.parameters()

    // maybe null if not weights in this module.
    require(weightParameters != null && weightParameters.length > 0,
      s"model ${this.getName} doesn't have any trainable parameters.")

    // If some gradParameters did not allocated storage, allocate it
    require(weightParameters.length == gradParameters.length,
      "weights and gradient number are not match")
    weightParameters.zip(gradParameters).foreach { case(w, g) =>
      g.resizeAs(w)
    }
    (Module.flatten[T](weightParameters), Module.flatten[T](gradParameters))
  }
}
