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
      val evalStats = mln.eval(readData(testFileName), conf.verbose())
      println(evalStats)
    }
  }

  def readData(filename: String) = {
    import com.opencsv.CSVReader
    import java.io._

    // Accumulator for the featurized items (label + feature vector).
    val data = new ListBuffer[(INDArray,INDArray)]()
    val iterator = new CSVReader(new FileReader(filename), ',', '"').iterator
    while (iterator.hasNext) {
      val items = iterator.next
      val labelString :: features = items.toList
      val label =
        if (labelString=="0") Array(1.,0.) // negative class
        else Array(0.,1.) // positive class
      data.append((Nd4j.create(label),Nd4j.create(features.map(_.toDouble).toArray)))
    }
    data.toList
  }
  
  def train(items: List[(INDArray,INDArray)], numLayers: Int) = {

    // Import convenience functions for creating MNL configs.
    import NetworkConfiguration._
    
    // We are using the word vectors as the input features, so obtain
    // the length of the vectors to set thu inputNum value to the MLN.
    val inputNum = items.head._2.length

    // Some parameters to set. 
    val outputNum = 2 // Because we are doing binary positive/negative classification.
    val iterations = 5
    val seed = 123

    // Train the model.
    log.info("Train model...")
    val mlnConf = {
      if (numLayers==2)
        twoLayerConf(seed, iterations, inputNum, outputNum)
      else
        oneLayerConf(seed, iterations, inputNum, outputNum)
    }
    
    val model = new MultiLayerNetwork(mlnConf)
    model.init()
    for ((labelRow, featureRow) <- items)
      model.fit(featureRow, labelRow)
    model
  }
  
}



