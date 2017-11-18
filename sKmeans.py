from pyspark.sql.types import *
from pyspark.ml.clustering import KMeans
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover

sc = SparkContext()
spark = SparkSession(sc)

#Collect text and text path
#Values are stored in key, value pair. text path being the key and text content being value 
documents = sc.wholeTextFiles("hdfs:///user/pcanogo/test3/")

#Construct schema to have a dataframe structure
#This creates a structure of col1 being text path and col2 being text content
schema =  StructType([StructField ("path" , StringType(), True) ,  StructField("text" , StringType(), True)])
docDataFrame = spark.createDataFrame(documents,schema)

#Tokenizing the words to make text string into text list
tokenizer = Tokenizer(inputCol="text", outputCol="terms")
wordsData = tokenizer.transform(docDataFrame)

#Removing stop words that don't bring any value
remover = StopWordsRemover(inputCol="terms", outputCol="filtered terms")
filteredData = remover.transform(wordsData)

#Count the term frequency 
hashingTF = HashingTF(inputCol="filtered terms", outputCol="term frequency", numFeatures=20)
featurizedData = hashingTF.transform(filteredData)

#Calculate the inverse document frequency
idf = IDF(inputCol="term frequency", outputCol="features", minDocFreq=1)
#create idf model and fit the data into shape
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

#Create Kmeans model
kmeans = KMeans(k=2)
kmeansModel = kmeans.fit(rescaledData)
predictionData = kmeansModel.transform(rescaledData)

#show full dataframe
predictionData.show()
#show only text name and cluster prediction
predictionData.select("path", "prediction").show(truncate=False)
