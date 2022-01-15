package com.mass.sparkdl.nn.abstractnn

import scala.reflect.ClassTag

import com.mass.sparkdl.nn.graphnn.Graph.ModuleNode
import com.mass.sparkdl.nn.graphnn.Edge
import com.mass.sparkdl.nn.mixin.Module
import com.mass.sparkdl.optim.OptimMethod
import com.mass.sparkdl.tensor.{Tensor, TensorDataType, TensorNumeric}
import com.mass.sparkdl.utils.DistriParameterSynchronizer
import org.apache.commons.lang3.SerializationUtils

abstract class AbstractModule[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag](
    implicit ev: TensorNumeric[T]) extends Serializable {

  var output: B = Activity.allocate[B, T]()
  var gradInput: A = Activity.allocate[A, T]()

  final def forward(input: A): B = {
    try {
      updateParameter()
      updateOutput(input)
    } catch {
      case e: Throwable =>
        throw e
    }
    output
  }

  def backward(input: A, gradOutput: B): A = {
    updateGradInput(input, gradOutput)
    accGradParameters(input, gradOutput)
    asyncGradient()
    gradInput
  }

  private[sparkdl] def asyncGradient(): Unit = {
    if (this.getParameterSynchronizer != null) {
      if (this.parameters() != null) {
        this.getParameterSynchronizer.put(this.getName)
      }
    }
  }

  def updateOutput(input: A): B

  def updateGradInput(input: A, gradOutput: B): A

  def accGradParameters(input: A, gradOutput: B): Unit = { }

  def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = null

  def getExtraParameter: Array[Tensor[T]] = null

  def setExtraParameter(extraParam: Array[Tensor[T]]): this.type = {
    val currentExtraParam = this.getExtraParameter
    if (extraParam != null && currentExtraParam != null) {
      require(extraParam.length == currentExtraParam.length)
      var i = 0
      while (i < extraParam.length) {
        currentExtraParam(i).copy(extraParam(i))
        i += 1
      }
      this
    } else if (extraParam == null && currentExtraParam == null) {
      this
    } else {
      throw new IllegalArgumentException
    }
  }

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

  private var _parameterSynchronizer: DistriParameterSynchronizer[T] = _

  private[sparkdl] def setParameterSynchronizer(
      parameterSynchronizer: DistriParameterSynchronizer[T]): Unit = {
    _parameterSynchronizer = parameterSynchronizer
  }

  private[sparkdl] def getParameterSynchronizer: DistriParameterSynchronizer[T] = {
    _parameterSynchronizer
  }

  private var _optimMethod: OptimMethod[T] = _

  private[sparkdl] def setOptimMethod(optimMethod: OptimMethod[T]): Unit = {
    _optimMethod = optimMethod
  }

  private[sparkdl] def getOptimMethod: OptimMethod[T] = _optimMethod

  private[sparkdl] def updateParameter(): Unit = {
    if (this.getParameterSynchronizer != null && this.isTraining) {
      if (this.parameters() != null) {
        val (weights, grads) = getParameterSynchronizer.get(this.getName)
        if (grads != null) {
          val optimMethod = this.getOptimMethod
          require(optimMethod != null)
          optimMethod.optimize(_ => (ev.fromType(0.0f), grads), weights)
          this.zeroGradParameters()
        }
      }
    }
  }

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
