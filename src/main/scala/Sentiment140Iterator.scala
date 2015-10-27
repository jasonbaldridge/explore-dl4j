package expdl4j

import com.opencsv.CSVReader
import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer
import java.io._
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor
import org.deeplearning4j.text.sentenceiterator.BaseSentenceIterator

/**
  * A class that allows one to iterate over items in the Sentiment140 data, and converts
  * each line to [1,0] for negative items and [0,1] for positive items plus the
  * preprocessed version of each document.
  */
class Sentiment140Iterator(
  filename: String,
  preprocessor: SentencePreProcessor = new SimplePreprocessor()
) extends BaseSentenceIterator {

  // CSV iterator over the document. It's a var so that we can reset it (to conform
  // to the BaseSentenceIterator interface).
  var iterator = new CSVReader(new FileReader(filename), ',', '"').iterator

  /** Does this iterator have more stuff? */
  def hasNext = iterator.hasNext

  /** Get the next sentence (ignores the label). */
  def nextSentence = preprocessor.preProcess(iterator.next.apply(5))

  /** Get the next label and sentence. */
  def nextLabelAndSentence: Option[(Array[Double],Array[String])] = {
    val info = iterator.next
    val integerLabel = info(0).toInt
    val labelsOpt =
      if (integerLabel==0) Some(Array(1.,0.)) // Negative sentiment
      else if (integerLabel==4) Some(Array(0.,1.)) // Positive sentiment
      else None
    val processedSentence = preprocessor.preProcess(info(5)).split("\\s+")
    labelsOpt.map(labels => (labels, processedSentence))
  }

  /** Reset the iterator. */
  def reset {
    iterator = new CSVReader(new FileReader(filename), ',', '"').iterator
  }
  
}
