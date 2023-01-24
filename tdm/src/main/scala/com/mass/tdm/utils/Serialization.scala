package com.mass.tdm.utils

import java.io._
import java.text.{DecimalFormat, NumberFormat}

import scala.util.{Failure, Success, Using}

import com.mass.scalann.Module
import com.mass.scalann.utils.{FileReader => DistFileReader, FileWriter => DistFileWriter}
import com.mass.tdm.operator.TDMOp
import org.apache.hadoop.io.IOUtils

object Serialization {

  def saveEmbeddings[T](path: String, model: Module[T], embedSize: Int): Unit = {
    val numItems = (math.pow(2, TDMOp.tree.getMaxLevel + 1) - 1).toInt
    val embedTensor = model.parameters()._1.head
    require(
      embedTensor.size().sameElements(Array(numItems, embedSize)),
      s"embedding size doesn't match, " +
        s"require: ${Array(numItems, embedSize).mkString("Array(", ", ", ")")}, " +
        s"found: ${embedTensor.size().mkString("Array(", ", ", ")")}"
    )

    val embeddings = embedTensor.storage().array()
    val idCodeMap = TDMOp.tree.getIdCodeMap
    val itemIds = idCodeMap.keys.toArray.sorted
    // val formatter = new DecimalFormat("###.######")
    val formatter = NumberFormat.getNumberInstance().asInstanceOf[DecimalFormat]
    // formatter.setMaximumFractionDigits(6)
    formatter.applyPattern("###.############")

    val fileWriter = DistFileWriter(path)
    val output = fileWriter.create(overwrite = true)
    Using(new BufferedWriter(new OutputStreamWriter(output))) { writer =>
      itemIds.foreach { id =>
        writer.write(s"$id")
        val code = idCodeMap(id)
        val offset = code * embedSize
        val end = offset + embedSize
        var i = offset
        while (i < end) {
          writer.write(s", ${formatter.format(embeddings(i))}")
          i += 1
        }
        writer.write('\n')
      }
    } match {
      case Success(_) =>
        output.close()
        fileWriter.close()
      case Failure(e: FileNotFoundException) =>
        println(s"""file "$path" not found""")
        throw e
      case Failure(t: Throwable) =>
        throw t
    }
  }

  def saveModel[T](path: String, model: Module[T]): Unit = {
    val fileWriter = DistFileWriter(path)
    val output: OutputStream = fileWriter.create(overwrite = true)
    val byteArrayOut = new ByteArrayOutputStream()
    val writer = new ObjectOutputStream(byteArrayOut)
    try {
      writer.writeObject(model)
      IOUtils.copyBytes(new ByteArrayInputStream(byteArrayOut.toByteArray), output, 1024, true)
    } catch {
      case e: FileNotFoundException =>
        println(s"""file "$path" not found""")
        throw e
      case t: Throwable =>
        throw t
    } finally {
      writer.close()
      byteArrayOut.close()
      output.close()
      fileWriter.close()
    }
  }

  def loadModel[T](path: String): Module[T] = {
    var model: Module[T] = null
    val fileReader = DistFileReader(path)
    val input = fileReader.open()
    val reader = new ObjectInputStream(new BufferedInputStream(input))
    try {
      model = reader.readObject().asInstanceOf[Module[T]]
    } catch {
      case e: FileNotFoundException =>
        println(s"""file "$path" not found""")
        throw e
      case t: Throwable =>
        throw t
    } finally {
      reader.close()
      input.close()
      fileReader.close()
    }
    model
  }

  def saveMapping(path: String, mapping: Map[Int, Int]): Unit = {
    Using.resource(new BufferedWriter(new FileWriter(path))) { writer =>
      mapping.foreach { case (item, id) =>
        writer.write(s"$item $id\n")
      }
    }
  }

  def loadMapping(path: String): Map[Int, Int] = {
    val fileSource = scala.io.Source.fromFile(path)
    fileSource
      .getLines()
      .map { line =>
        val kv = line.split("\\s+").map(_.trim)
        kv.head.toInt -> kv.last.toInt
      }
      .toMap
  }

  def loadBothMapping(path: String): (Map[Int, Int], Map[Int, Int]) = {
    val itemIdMapping = loadMapping(path)
    (itemIdMapping, itemIdMapping.map(_.swap))
  }
}
