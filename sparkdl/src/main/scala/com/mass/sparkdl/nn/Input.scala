package com.mass.sparkdl.nn

import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.{AbstractModule, Activity}
import com.mass.sparkdl.nn.Graph.ModuleNode
import com.mass.sparkdl.tensor.TensorNumeric
import com.mass.sparkdl.utils.Node

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

object Input{
  def apply[T: ClassTag](name: String = null)(implicit ev: TensorNumeric[T]): ModuleNode[T] = {
    val module = new Input()
    if (name != null) {
      module.setName(name)
    }
    new Node(module.asInstanceOf[AbstractModule[Activity, Activity, T]])
  }
}
