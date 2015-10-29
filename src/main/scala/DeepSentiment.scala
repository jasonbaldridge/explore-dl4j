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
  * A container for a trained MLN to use on future predictions.
  */
class DeepSentiment(model: MultiLayerNetwork) {
  
  def eval(items: List[(INDArray,INDArray)], verbose: Boolean = false) = {
    val eval = new Evaluation()
    for ((labelRow, featureRow) <- items) {
      val output = model.output(featureRow)

      if (verbose)
        println(labelRow + "\t" + output)

      eval.eval(labelRow, output)
    }
    eval.stats
  }
  
}


/**
  * Static methods for training MLN to construct a reusable sentiment classifier.
  */
object DeepSentiment {

  import Word2VecUtil._

  lazy val log = LoggerFactory.getLogger(expdl4j.DeepSentiment.getClass)

  /**
    * Main method: train word vectors if necessary. Then use those to transform base tweets into
    * dense vectors by averaging vectors for all words. Train the model with those and
    * create a DeepSentiment instance which can be used for eval.
    */ 
  def main(args: Array[String]) {
    val conf = new ClassifierCommand(args)
    conf.afterInit()
    
    // Load word vectors.
    val wordVectors = getVectors(conf.vectorfile())
    val w2vUtil = new Word2VecUtil(wordVectors)
    
    // Train the model.
    val trainingData = vectorizeData(conf.trainfile(),w2vUtil)
    val model = train(trainingData, conf.numlayers())
  
    // Construct a reusable DeepSentiment object that can eval new items.
    val deepSentiment = new DeepSentiment(model)
  
    // Evaluate on test data if provided.
    conf.evalfile.get.foreach { testFileName =>
      val evalStats = deepSentiment.eval(vectorizeData(testFileName,w2vUtil), conf.verbose())
      println(evalStats)
    }
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

  
  def train(items: List[(INDArray,INDArray)], numLayers: Int) = {

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

  /**
    * A configuration for doing logistic regression, though it seems to do so poorly.
    * Not sure what to expect exactly -- could be the linear algebra or some other factor?
    */
  def oneLayerConf(seed: Int, iterations: Int, inputNum: Int, outputNum: Int) = {
    new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .learningRate(1e-3)
      .l1(0.3).regularization(true).l2(1e-3)
      .list(1)
      .layer(0, new OutputLayer.Builder().activation("softmax").nIn(inputNum).nOut(outputNum).build())
      .build()

  }

  /**
    * An MLN configuration for a single hidden layer. This was adapted from DBNIrisExample in
    * the DL4J scala examples.
    */
  def twoLayerConf(seed: Int, iterations: Int, inputNum: Int, outputNum: Int, layer1Num: Int = 10) = {
    new NeuralNetConfiguration.Builder()
      .seed(seed) // Locks in weight initialization for tuning
      .iterations(iterations) // # training iterations predict/classify & backprop
      .learningRate(1e-6f) // Optimization step size
      .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT) // Backprop to calculate gradients
      .l1(1e-1).regularization(true).l2(2e-4)
      .useDropConnect(true)
      .list(2) // # NN layers (doesn't count input layer)
      .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
        .nIn(inputNum) // # input nodes
        .nOut(layer1Num) // # fully connected hidden layer nodes. Add list if multiple layers.
        .weightInit(WeightInit.XAVIER) // Weight initialization
        .k(1) // # contrastive divergence iterations
        .activation("relu") // Activation function type
        .lossFunction(LossFunctions.LossFunction.RMSE_XENT) // Loss function type
        .updater(Updater.ADAGRAD)
        .dropOut(0.5)
        .build()
    ) // NN layer type
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .nIn(layer1Num) // # input nodes
        .nOut(outputNum) // # output nodes
        .activation("softmax")
        .build()
    ) // NN layer type
      .build()
  }
  
}



