package com.mass.scalann.nn

import scala.reflect.ClassTag

import com.mass.scalann.nn.abstractnn.{AbstractModule, Activity}
import com.mass.scalann.nn.graphnn.Graph.ModuleNode
import com.mass.scalann.nn.graphnn.Node
import com.mass.scalann.tensor.TensorNumeric

class Input[T: ClassTag]()(implicit ev: TensorNumeric[T])
    extends AbstractModule[Activity, Activity, T] {

  override def updateOutput(input: Activity): Activity = {
    output = input
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput = gradOutput
    gradInput
  }
}

object Input {
  def apply[T: ClassTag](name: String = null)(implicit ev: TensorNumeric[T]): ModuleNode[T] = {
    val module = new Input()
    if (name != null) {
      module.setName(name)
    }
    new Node(module.asInstanceOf[AbstractModule[Activity, Activity, T]])
  }
}
