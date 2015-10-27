name := "Explore DL4J"

version := "0.1-SNAPSHOT"

scalaVersion := "2.10.4"

lazy val root = project.in(file("."))

libraryDependencies ++= Seq(
  "org.rogach" %% "scallop" % "0.9.5",
  "com.opencsv" % "opencsv" % "3.4",
  "commons-io" % "commons-io" % "2.4",
  "com.google.guava" % "guava" % "18.0",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc3.4",
  "org.deeplearning4j" % "deeplearning4j-nlp" % "0.4-rc3.4",
  "org.deeplearning4j" % "deeplearning4j-ui" % "0.4-rc3.4",
  "org.jblas" % "jblas" % "1.2.4",
  "org.nd4j" % "canova-nd4j-image" % "0.0.0.11",
  "org.nd4j" % "nd4j-jblas" % "0.4-rc3.5",
  "org.nd4j" % "nd4j-x86" % "0.4-rc3.5"
)

resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

resolvers += "Sonatype release Repository" at "http://oss.sonatype.org/service/local/staging/deploy/maven2/"

resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository/"

enablePlugins(JavaAppPackaging)

mainClass in Compile := Some("expdl4j.DeepSentiment")
