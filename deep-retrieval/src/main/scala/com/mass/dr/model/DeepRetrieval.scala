package com.mass.dr.model

import java.io.{BufferedInputStream, ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}

import com.mass.dr.{paddingIdx, sigmoid}
import com.mass.scalann.utils.{FileReader => DistFileReader, FileWriter => DistFileWriter}
import org.apache.hadoop.io.IOUtils

class DeepRetrieval(
    val layerModel: LayerModel,
    val reRankModel: RerankModel,
    numItem: Int,
    numNode: Int,
    numLayer: Int,
    seqLen: Int,
    embedSize: Int) extends Serializable with CandidateSearcher {

  def recommend(
    sequence: Seq[Int],
    topk: Int,
    beamSize: Int,
    mappings: MappingOp
  ): Seq[(Int, Double)] = {
    val sequenceIds = sequence.map(mappings.itemIdMapping.getOrElse(_, paddingIdx))
    val candidateItems = searchCandidate(
      sequenceIds,
      layerModel,
      beamSize,
      mappings.pathItemMapping
    )
    val reRankScores = reRankModel.inference(candidateItems, sequenceIds)

    candidateItems
      .zip(reRankScores)
      .sortBy(_._2)(Ordering[Double].reverse)
      .take(topk)
      .map(i => (mappings.idItemMapping(i._1), sigmoid(i._2)))
  }
}

object DeepRetrieval {

  def apply(
    layerModel: LayerModel,
    reRankModel: RerankModel,
    numItem: Int,
    numNode: Int,
    numLayer: Int,
    seqLen: Int,
    embedSize: Int
  ): DeepRetrieval = {
    new DeepRetrieval(
      layerModel,
      reRankModel,
      numItem,
      numNode,
      numLayer,
      seqLen,
      embedSize
    )
  }

  def saveModel(model: DeepRetrieval, modelPath: String): Unit = {
    val fileWriter = DistFileWriter(modelPath)
    val output = fileWriter.create(overwrite = true)
    val byteArrayOut = new ByteArrayOutputStream()
    val writer = new ObjectOutputStream(byteArrayOut)
    try {
      writer.writeObject(model)
      IOUtils.copyBytes(new ByteArrayInputStream(byteArrayOut.toByteArray), output, 1024, true)
    } catch {
      case t: Throwable =>
        throw t
    } finally {
      writer.close()
      byteArrayOut.close()
      output.close()
      fileWriter.close()
    }
  }

  def loadModel(modelPath: String): DeepRetrieval = {
    var model: DeepRetrieval = null
    val fileReader = DistFileReader(modelPath)
    val input = fileReader.open()
    val reader = new ObjectInputStream(new BufferedInputStream(input))
    try {
      model = reader.readObject().asInstanceOf[DeepRetrieval]
    } catch {
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
