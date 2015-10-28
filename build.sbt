name := "Explore DL4J"

version := "0.1-SNAPSHOT"

scalaVersion := "2.10.4"

lazy val dl4jVersion = "0.4-rc3.4"

lazy val nd4jVersion = "0.4-rc3.5"

libraryDependencies ++= Seq(
  "org.rogach" %% "scallop" % "0.9.5",
  "com.opencsv" % "opencsv" % "3.4",
  "commons-io" % "commons-io" % "2.4",
  "com.google.guava" % "guava" % "18.0",
  "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion,
  "org.deeplearning4j" % "deeplearning4j-nlp" % dl4jVersion,
  "org.deeplearning4j" % "deeplearning4j-ui" % dl4jVersion,
  "org.jblas" % "jblas" % "1.2.4",
  "org.nd4j" % "canova-nd4j-image" % "0.0.0.11",
  "org.nd4j" % "nd4j-jblas" % nd4jVersion
)

resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

resolvers += "Sonatype release Repository" at "http://oss.sonatype.org/service/local/staging/deploy/maven2/"

resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository/"

enablePlugins(JavaAppPackaging)

mainClass in Compile := Some("expdl4j.DeepSentiment")
