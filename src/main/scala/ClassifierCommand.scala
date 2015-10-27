package expdl4j

import org.rogach.scallop._

class ClassifierCommand(arguments: Seq[String]) extends ScallopConf(arguments) {

  // Labeled training examples.
  lazy val trainfile = opt[String]()

  // Labeled evaluation examples.
  lazy val evalfile = opt[String]()

  // File containing word2vec vectors.
  lazy val vectorfile = opt[String]()

  // The maximum number of training examples to use. If unspecified,
  // all are used.
  lazy val maxtraining = opt[Int]()

  // Configuration
  lazy val numlayers = opt[Int](default=Some(1))
  
  // Show verbose output.
  lazy val verbose = opt[Boolean](default = Some(false))

}
