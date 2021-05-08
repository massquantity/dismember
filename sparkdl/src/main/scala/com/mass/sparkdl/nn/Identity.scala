package com.mass.sparkdl.nn

import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}

class Identity[T: ClassTag]()(implicit ev: TensorNumeric[T])
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

object Identity {
  def apply[@specialized(Float, Double) T: ClassTag]()(
      implicit ev: TensorNumeric[T]): Identity[T] = {
    new Identity[T]()
  }
}
