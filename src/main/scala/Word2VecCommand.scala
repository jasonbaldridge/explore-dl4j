package expdl4j

import org.rogach.scallop._

class Word2VecCommand(arguments: Seq[String]) extends ScallopConf(arguments) {

  // Labeled training examples.
  lazy val trainFile = opt[String](descr="The file containing training sentences.", required=true)

  // File to output word2vec vectors.
  lazy val outputFile = opt[String](descr="The filename to output word vectors to.", required=true)

  // The length of the w2v vectors.
  lazy val numDimensions = opt[Int](
    default=Some(200),
    descr="The dimensionality of the vectors.",
    validate = 0<
  )
    

  // The input type.
  lazy val inputType = opt[String](
    default=Some("raw"),
    descr="The type of input containing sentences. (raw = one sentence per line, sentiment140 = lines from the sentiment140 data.)",
    validate=Set("raw","sentiment140"))

  // Show verbose output.
  lazy val verbose = opt[Boolean](descr="Be verbose.", default = Some(false))

}
