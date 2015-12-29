package expdl4j

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{ OutputLayer, RBM }
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.nn.conf.{ MultiLayerConfiguration, NeuralNetConfiguration, Updater }

import scala.collection.mutable.ListBuffer
import java.util.ArrayList
import java.io.File

/**
  * Static methods for training MLNs with preprocessed input features.
  */
object SimpleClassifier {

  lazy val log = LoggerFactory.getLogger(expdl4j.SimpleClassifier.getClass)

  /**
    * Main method: train word vectors if necessary. Then use those to transform base tweets into
    * dense vectors by averaging vectors for all words. Train the model with those and
    * create a DeepSentiment instance which can be used for eval.
    */ 
  def main(args: Array[String]) {
    val conf = new ClassifierCommand(args)
    conf.afterInit()

    // Train the model.
    val model = train(readData(conf.trainFile()), conf.numLayers())
  
    // Construct a reusable MLN container that can eval new items.
    val mln = new MLNContainer(model)
  
    // Evaluate on test data if provided.
    conf.evalFile.get.foreach { testFileName =>
      val evalItems = readData(testFileName).take(10)
      val (labels,features) = evalItems.unzip
      val ndlabels = Nd4j.create(labels.toArray)
      val ndfeatures = Nd4j.create(features.toArray)
      val evalStats = mln.eval(List((ndlabels,ndfeatures)), conf.verbose())
      println(evalStats)
      //for ((labelRow, featureRow) <- evalItems.take(5)) {
      //  val feats = Nd4j.create(featureRow)
      //  val pred = model.predict(feats)
      val output = model.output(ndfeatures)
      println("Feat: " + ndfeatures)
      //  println("Gold: " + labelRow.mkString(","))
      println("Out:  " + output)
      //  println("Pred: " + pred.mkString(","))
      //  println
      //}
    }
    println(model.params)
  }

  def readData(filename: String) = {
    import com.opencsv.CSVReader
    import java.io._

    // Accumulator for the featurized items (label + feature vector).
    val data = new ListBuffer[(Array[Double],Array[Double])]()
    val iterator = new CSVReader(new FileReader(filename), ',', '"').iterator
    while (iterator.hasNext) {
      val items = iterator.next
      val labelString :: features = items.toList
      val label =
        if (labelString=="0") Array(0.0,1.0) // negative class
        else Array(1.0,0.0) // positive class
      data.append((label,features.map(_.toDouble).toArray))
    }
    data.toList
  }
  
  def train(items: List[(Array[Double],Array[Double])], numLayers: Int) = {

    // Import convenience functions for creating MNL configs.
    import NetworkConfiguration._
    
    // We are using the word vectors as the input features, so obtain
    // the length of the vectors to set the inputNum value to the MLN.
    val inputNum = items.head._2.length

    // Some parameters to set. 
    val outputNum = 2 // Because we are doing binary positive/negative classification.
    val iterations = 50
    val seed = 123

    // Train the model.
    log.info("Train model...")
    val mlnConf = {
      if (numLayers==2) {
        // Constrain the layer one dimension to be 1/4 of the size of the input dimension.
        //val layer1Size = (inputNum/4).toInt
        val layer1Size = 4
        twoLayerConf(seed, iterations, inputNum, outputNum, layer1Size)
      } else {
        oneLayerConf(seed, iterations, inputNum, outputNum)
      }
    }
    
    val model = new MultiLayerNetwork(mlnConf)
    model.init()
    for (batch <- items.grouped(50)) {
      val (batchLabels, batchFeatures) = batch.unzip
      val ndlabels = Nd4j.create(batchLabels.toArray)
      val ndfeatures = Nd4j.create(batchFeatures.toArray)
      model.fit(ndfeatures,ndlabels)
    }
    model
  }
  
}
