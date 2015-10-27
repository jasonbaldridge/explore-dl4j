package expdl4j

import javafx.util.Pair
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
    val Array(trainFileName, testFileName, wordVectorFileName) = args

    // We'll set this in code for now, but could be configured.
    val vectorLength = 200
    
    // Train word vectors if they haven't been already.
    if (!new File(wordVectorFileName).isFile)
      trainAndSaveWord2Vec(trainFileName, wordVectorFileName, vectorLength)

    // Load trained vectors.
    val wordVectors = getVectors(wordVectorFileName)

    // Train the model.
    val model = train(computeAvgWordVector(trainFileName,wordVectors), vectorLength)

    // Construct a reusable DeepSentiment object that can eval new items.
    val deepSentiment = new DeepSentiment(model)

    // Evaluate on test data.
    val evalStats = deepSentiment.eval(computeAvgWordVector(testFileName,wordVectors))

    // Use this to see the output predictions on the first twenty items.
    //val evalStats = deepSentiment.eval(computeAvgWordVector(testFileName,wordVectors).take(20), true)

    // Use this to see the output predictions on the first twenty items of the training set.
    //val evalStats = deepSentiment.eval(computeAvgWordVector(trainFileName,wordVectors).take(20), true)

    println(evalStats)
  }

  def train(items: List[(INDArray,INDArray)], inputNum: Int) = {

    log.info("Build model....")

    // Some parameters to set. 
    val outputNum = 2 // Because we are doing binary positive/negative classification.
    val iterations = 5
    val seed = 123

    // Train the model.
    log.info("Train model...")
    val conf = singleLayerConf(seed, iterations, inputNum, outputNum)
    //val conf = twoLayerConf(seed, iterations, inputNum, outputNum)
    val model = new MultiLayerNetwork(conf)
    model.init()
    for ((labelRow, featureRow) <- items)
      model.fit(featureRow, labelRow)
    model
  }

  /**
    * A configuration for doing logistic regression, though it seems to do so poorly.
    * Not sure what to expect exactly -- could be the linear algebra or some other factor?
    */
  def singleLayerConf(seed: Int, iterations: Int, inputNum: Int, outputNum: Int) = {
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



