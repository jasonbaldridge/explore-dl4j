package expdl4j

import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor

/**
  * A class that preprocesses a document by lower casing it and throwing away all
  * non-alpha tokens.
  */
class SimplePreprocessor extends SentencePreProcessor {
  override def preProcess(s: String) = s.toLowerCase().replaceAll("[^a-zA-Z ]", "")
}
