package expdl4j

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer
import java.io._
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor
import org.deeplearning4j.text.sentenceiterator.BaseSentenceIterator
import org.nd4j.linalg.api.ndarray.INDArray

/**
  * A class that allows one to iterate over items in the Sentiment140 data, and converts
  * each line to [1,0] for negative items and [0,1] for positive items plus the
  * preprocessed version of each document.
  */
class RawSentenceIterator(
  filename: String,
  preprocessor: SentencePreProcessor = new SimplePreprocessor()
) extends BaseSentenceIterator {

  var iterator = scala.io.Source.fromFile(filename).getLines

  /** Does this iterator have more stuff? */
  def hasNext = iterator.hasNext

  /** Get the next sentence (ignores the label). */
  def nextSentence = preprocessor.preProcess(iterator.next)

  /** Reset the iterator. */
  def reset {
    iterator = scala.io.Source.fromFile(filename).getLines
  }
  
}
