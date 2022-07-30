package com.mass.scalann.nn

import scala.reflect.ClassTag

import com.mass.scalann.nn.abstractnn.{AbstractModule, Activity}
import com.mass.scalann.tensor.TensorNumeric

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
