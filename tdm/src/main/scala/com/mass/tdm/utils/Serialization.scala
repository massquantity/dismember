package com.mass.tdm.utils

import java.io._
import java.text.{DecimalFormat, NumberFormat}

import scala.util.{Failure, Success, Using}

import com.mass.sparkdl.Module
import com.mass.sparkdl.utils.{FileReader => DistFileReader, FileWriter => DistFileWriter}
import com.mass.tdm.operator.TDMOp
import org.apache.hadoop.io.IOUtils

object Serialization {

  def saveEmbeddings(path: String, model: Module[Float], embedSize: Int): Unit = {
    val numItems = (math.pow(2, TDMOp.tree.getMaxLevel) - 1).toInt
    val embedTensor = model.parameters()._1.head
    require(embedTensor.size().sameElements(Array(numItems, embedSize)),
      s"embedding size doesn't match, " +
        s"require: ${Array(numItems, embedSize).mkString("Array(", ", ", ")")}, " +
        s"found: ${embedTensor.size().mkString("Array(", ", ", ")")}")

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

  def saveModel(path: String, model: Module[Float]): Unit = {
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

  def loadModel(path: String): Module[Float] = {
    var model: Module[Float] = null
    val fileReader = DistFileReader(path)
    val input = fileReader.open()
    val reader = new ObjectInputStream(new BufferedInputStream(input))
    try {
      model = reader.readObject().asInstanceOf[Module[Float]]
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

}
