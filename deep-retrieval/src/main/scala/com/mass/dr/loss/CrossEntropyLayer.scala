package com.mass.dr.loss

import com.mass.scalann.nn.CrossEntropyCriterion
import com.mass.scalann.tensor.Tensor
import com.mass.scalann.tensor.TensorNumeric.NumericDouble
import com.mass.scalann.utils.Table
import org.apache.commons.lang3.SerializationUtils

class CrossEntropyLayer(numLayer: Int) extends Serializable {

  private val losses = Array.fill(numLayer)(CrossEntropyCriterion())

  def forward(input: Table, target: IndexedSeq[Tensor[Double]]): IndexedSeq[Double] = {
    (0 until numLayer).map { i =>
      losses(i).forward(input[Tensor[Double]](i), target(i))
    }
  }

  def backward(input: Table, target: IndexedSeq[Tensor[Double]]): Table = {
    val gradInput = (0 until numLayer).map { i =>
      losses(i).backward(input[Tensor[Double]](i), target(i))
    }
    Table.seq(gradInput)
  }

  override def clone(): CrossEntropyLayer = {
    SerializationUtils.clone(this)
  }
}

object CrossEntropyLayer {

  def apply(numLayer: Int): CrossEntropyLayer = {
    new CrossEntropyLayer(numLayer)
  }
}
