package expdl4j

import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.word2vec.Word2Vec
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
  * Static methods for training MLN to construct a reusable sentiment classifier.
  */
object DeepSentiment {

  import Word2VecUtil._

  lazy val log = LoggerFactory.getLogger(expdl4j.DeepSentiment.getClass)

  /**
    * Main method: Use trained word vectors those to transform base tweets into
    * dense vectors by averaging vectors for all words. Train the model with those and
    * create an MLNContainer instance which can be used for eval.
    */ 
  def main(args: Array[String]) {
    val conf = new ClassifierCommand(args)
    conf.afterInit()
    
    // Load word vectors.
    val wordVectors = getVectors(conf.vectorFile())
    val w2vUtil = new Word2VecUtil(wordVectors)
    
    // Train the model.
    val trainingData = vectorizeData(conf.trainFile(),w2vUtil)
    //val model = SimpleClassifier.train(trainingData, conf.numLayers())
    //
    //// Construct a reusable MLNContainer object that can eval new items.
    //val deepSentiment = new MLNContainer(model)
    //
    //// Evaluate on test data if provided.
    //conf.evalFile.get.foreach { testFileName =>
    //  val evalStats = deepSentiment.eval(vectorizeData(testFileName,w2vUtil), conf.verbose())
    //  println(evalStats)
    //}
  }

  /**
    * Given input documents and word vectors, compute vec-length representation of each
    * document for input to net.
    */
  def vectorizeData(inputFilename: String, w2vUtil: Word2VecUtil) = {

    // Accumulator for the featurized items (label + document as vector).
    val data = new ListBuffer[(INDArray,INDArray)]()

    // Parse the csv file again to get label and average word vector
    val it = new Sentiment140Iterator(inputFilename)
    while (it.hasNext()) {

      // The iterator returns an option. This enables a clean solution
      // to skipping neutral items for this particular task.
      it.nextLabelAndSentence.map { case(label,words) =>
          data.append((Nd4j.create(label),w2vUtil.vectorizeDocument(words)))
      }
    }
    
    data.toList
  }

}
