package com.mass.scalann.optim

import com.mass.scalann.utils.Table

trait Trigger extends Serializable {
  def apply(state: Table): Boolean
}

object Trigger {

  def maxIteration(max: Int, name: String): Trigger = {
    new Trigger() {
      override def apply(state: Table): Boolean = {
        state[Int](name) >= max
      }
    }
  }
}
