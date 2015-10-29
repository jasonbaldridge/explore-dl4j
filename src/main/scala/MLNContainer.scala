package expdl4j

import org.deeplearning4j.eval.Evaluation
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

/**
  * A container for a trained MLN to use on future predictions.
  */
class MLNContainer(model: MultiLayerNetwork) {
  
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
