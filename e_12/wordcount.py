import string, re
import sys
from pyspark.sql import SparkSession,functions, types, Row
from pyspark.sql.functions import col, explode, split, lower, count

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 10)
assert spark.version >= '3.5'

input= sys.argv[1]
output = sys.argv[2]

wordbreak = r"[%s\s]+" % re.escape(string.punctuation)

lines_df = spark.read.text(input)

words_df = lines_df.select(explode(split(lower(col("value")), wordbreak)).alias("word"))

words_df = words_df.filter(col("word") != "")
word_counts_df = words_df.groupBy("word").agg(count("word").alias("count"))


sorted_counts= word_counts_df.orderBy(col("count").desc(), col("word").asc())

sorted_counts.write.csv(output, header=True, mode="overwrite")

