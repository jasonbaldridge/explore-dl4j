package expdl4j

import org.rogach.scallop._

class ClassifierCommand(arguments: Seq[String]) extends ScallopConf(arguments) {

  // Labeled training examples.
  lazy val trainFile = opt[String](descr="Training examples.", required=true)

  // Labeled evaluation examples.
  lazy val evalFile = opt[String](descr="Evaluation examples.")

  // File containing word2vec vectors.
  lazy val vectorFile = opt[String](descr="File containing word vectors, if needed.")

  // Configuration
  lazy val numLayers = opt[Int](default=Some(1),validate=Set(1,2))
  
  // Show verbose output.
  lazy val verbose = opt[Boolean](default = Some(false))

}
