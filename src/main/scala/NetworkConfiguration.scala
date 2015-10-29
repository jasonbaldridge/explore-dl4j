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

object NetworkConfiguration {
  
  /**
    * A configuration for doing logistic regression.
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



