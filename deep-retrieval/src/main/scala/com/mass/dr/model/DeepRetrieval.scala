package com.mass.dr.model

import java.io.{BufferedInputStream, ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}

import com.mass.dr.dataset.LocalDataSet.loadMapping
import com.mass.dr.{LayerModule, Path, RerankModule}
import com.mass.dr.model.RerankModel.{inference => rerankCandidates}
import com.mass.dr.dataset.LocalDataSet
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble
import com.mass.sparkdl.utils.{FileReader => DistFileReader, FileWriter => DistFileWriter}
import org.apache.hadoop.io.IOUtils

class DeepRetrieval(
    numItem: Int,
    numNode: Int,
    numLayer: Int,
    seqLen: Int,
    embedSize: Int,
    paddingIdx: Int) extends Serializable with CandidateSearcher {
  import DeepRetrieval._

  @transient private var itemIdMapping: Map[Int, Int] = _
  @transient private var itemPathMapping: Map[Int, Seq[Path]] = _
  @transient lazy val idItemMapping: Map[Int, Int] = itemIdMapping.map(i => i._2 -> i._1)
  @transient lazy val pathItemsMapping: Map[Path, Seq[Int]] = {
    itemPathMapping
      .flatMap { case (item, paths) => paths.map((_, item)) }
      .groupBy(_._1)
      .map(i => i._1 -> i._2.values.toSeq)
  }

  val layerModel: Seq[LayerModule[Double]] = LayerModel.buildModel(
    numItem,
    numNode,
    numLayer,
    seqLen,
    embedSize,
    paddingIdx
  )
  val reRankModel: RerankModule[Double] = RerankModel.trainModel(
    numItem,
    seqLen,
    embedSize,
    paddingIdx
  )
  val reRankWeights: Tensor[Double] = Tensor[Double](numItem, seqLen * embedSize).randn(0.0, 0.05)
  val reRankBias: Tensor[Double] = Tensor[Double](numItem).zero()

  def recommend(sequence: Seq[Int], topk: Int, beamSize: Int): Seq[(Int, Double)] = {
    val sequenceIds = sequence.map(itemIdMapping.getOrElse(_, paddingIdx))
    val candidateItems = searchCandidate(
      sequenceIds,
      layerModel,
      beamSize,
      pathItemsMapping
    )
    val reRankScores = rerankCandidates(
      candidateItems,
      sequenceIds,
      reRankModel,
      reRankWeights,
      reRankBias,
      seqLen,
      embedSize
    ).storage().array()

    candidateItems
      .zip(reRankScores)
      .sortBy(_._2)(Ordering[Double].reverse)
      .take(topk)
      .map(i => (idItemMapping(i._1), sigmoid(i._2)))
  }

  def setMapping(mappingPath: String): this.type = {
    val tmp = loadMapping(mappingPath)
    itemIdMapping = tmp._1
    itemPathMapping = tmp._2
    this
  }

  def setMapping(dataset: LocalDataSet): Unit = {
    itemIdMapping = dataset.itemIdMapping
    itemPathMapping = dataset.itemPathMapping
  }
}

object DeepRetrieval {

  def apply(
      numItem: Int,
      numNode: Int,
      numLayer: Int,
      seqLen: Int,
      embedSize: Int,
      paddingIdx: Int): DeepRetrieval = {
    new DeepRetrieval(
      numItem,
      numNode,
      numLayer,
      seqLen,
      embedSize,
      paddingIdx
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

  def loadModel(modelPath: String, mappingPath: String): DeepRetrieval = {
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
    model.setMapping(mappingPath)
  }

  def sigmoid(logit: Double): Double = {
    1.0 / (1 + java.lang.Math.exp(-logit))
  }
}
