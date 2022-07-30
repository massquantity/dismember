package com.mass.dr.model

import java.io.{BufferedInputStream, BufferedOutputStream, DataInputStream, OutputStream}
import java.nio.ByteBuffer

import scala.collection.mutable
import scala.util.{Failure, Random, Success, Using}

import com.mass.dr.{Path => DRPath}
import com.mass.dr.model.MappingOp.pathToItems
import com.mass.dr.protobuf.item_mapping.{ItemSet, Item => ProtoItem, Path => ProtoPath}
import com.mass.scalann.utils.{FileReader => DistFileReader, FileWriter => DistFileWriter}

case class MappingOp(itemIdMapping: Map[Int, Int], itemPathMapping: Map[Int, Seq[DRPath]]) {
  val idItemMapping: Map[Int, Int] = itemIdMapping.map(i => i._2 -> i._1)

  val pathItemMapping: Map[DRPath, Seq[Int]] = pathToItems(itemPathMapping)
}

object MappingOp {

  def pathToItems(itemPathMapping: Map[Int, Seq[DRPath]]): Map[DRPath, Seq[Int]] = {
    itemPathMapping
      .flatMap { case (item, paths) => paths.map((_, item)) }
      .groupBy(_._1)
      .map(i => i._1 -> i._2.values.toSeq)
  }

  def initItemPathMapping(
    numItem: Int,
    numLayer: Int,
    numNode: Int,
    numPathPerItem: Int
  ): Map[Int, Seq[DRPath]] = {
    (0 until numItem).map(_ -> {
      val m = for {
        _ <- 1 to numPathPerItem
        _ <- 1 to numLayer
      } yield Random.nextInt(numNode)
      m.sliding(numLayer, numLayer).toSeq
    }).toMap
  }

  def writeMapping(
    outputPath: String,
    itemIdMapping: Map[Int, Int],
    itemPathMapping: Map[Int, Seq[DRPath]]
  ): Unit = {
    val allItems = ItemSet(
      itemIdMapping.map { case (item, id) =>
        val paths = itemPathMapping(id).map(ProtoPath(_))
        ProtoItem(item, id, paths)
      }.toSeq
    )

    val fileWriter = DistFileWriter(outputPath)
    val output: OutputStream = fileWriter.create(overwrite = true)
    Using(new BufferedOutputStream(output)) { writer =>
      val bf: ByteBuffer = ByteBuffer.allocate(4).putInt(allItems.serializedSize)
      writer.write(bf.array())
      allItems.writeTo(writer)
    } match {
      case Success(_) =>
        output.close()
        fileWriter.close()
      case Failure(t: Throwable) =>
        throw t
    }
  }

  def loadMapping(path: String): (Map[Int, Int], Map[Int, Seq[DRPath]]) = {
    val idMapping = mutable.Map.empty[Int, Int]
    val pathMapping = mutable.Map.empty[Int, Seq[DRPath]]
    val fileReader = DistFileReader(path)
    val input = fileReader.open()
    Using(new DataInputStream(new BufferedInputStream(input))) { input =>
      val size = input.readInt()
      val buf = new Array[Byte](size)
      input.readFully(buf)
      val allItems = ItemSet.parseFrom(buf).items
      allItems.foreach { i =>
        idMapping(i.item) = i.id
        pathMapping(i.id) = i.paths.map(_.index.toIndexedSeq)
      }
    } match {
      case Success(_) =>
        input.close()
        fileReader.close()
      case Failure(t: Throwable) =>
        throw t
    }
    (Map.empty ++ idMapping, Map.empty ++ pathMapping)
  }

  def loadMappingOp(path: String): MappingOp = {
    val tmp = loadMapping(path)
    MappingOp(tmp._1, tmp._2)
  }
}
