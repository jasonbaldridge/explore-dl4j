package expdl4j

import org.rogach.scallop._

class Word2VecCommand(arguments: Seq[String]) extends ScallopConf(arguments) {

  // Labeled training examples.
  lazy val trainfile = opt[String]()

  // File to output word2vec vectors.
  lazy val vectorfile = opt[String]()

  // The length of the w2v vectors.
  lazy val vectorlength = opt[Int](default=Some(200))

  // Show verbose output.
  lazy val verbose = opt[Boolean](default = Some(false))

}
