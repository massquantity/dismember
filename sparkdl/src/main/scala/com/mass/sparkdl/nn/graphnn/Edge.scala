package com.mass.sparkdl.nn.graphnn

private[sparkdl] class Edge(val fromIndex: Option[Int]) extends Serializable {

  override def toString: String = {
    s"Edge(fromIndex: $fromIndex)"
  }

  def newInstance(): Edge = {
    fromIndex match {
      case Some(index) => Edge(index)
      case None => Edge()
    }
  }
}

object Edge {
  def apply(value: Int): Edge = new Edge(Some(value))

  def apply(): Edge = new Edge(None)
}
