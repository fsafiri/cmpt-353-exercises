import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('link_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('parent_id', types.StringType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('score_hidden', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('ups', types.LongType()),
    #types.StructField('year', types.IntegerType()),
    #types.StructField('month', types.IntegerType()),
])


def main(in_directory, out_directory):
    comments = spark.read.json(in_directory, schema=comments_schema)
    averages_by_subreddit = comments.groupBy('subreddit').agg(functions.avg('score').alias('average')).filter('average >= 0')


    comments = comments.join(averages_by_subreddit, (averages_by_subreddit.subreddit == comments.subreddit), how='inner').drop(averages_by_subreddit.subreddit)
   #broadcast hint 
    #comments = comments.join(averages_by_subreddit.hint("broadcast"), (averages_by_subreddit.subreddit == comments.subreddit), how='inner').drop(averages_by_subreddit.subreddit)
   
    filtered = comments.select(comments['subreddit'],comments['author'],(comments['score'] / comments['average']).alias('rel_score'))
    filtered = filtered.cache()

    grouped = filtered.groupBy(filtered['subreddit'])
    max_comments = grouped.agg(functions.max(filtered['rel_score']).alias('max_rel_score'))

    filtered = filtered.join(max_comments,(max_comments.subreddit == filtered.subreddit) & (max_comments.max_rel_score == filtered.rel_score),how='inner').drop(max_comments.subreddit).drop(max_comments.max_rel_score)
   #filtered = filtered.join(max_comments.hint("broadcast"), (max_comments.subreddit == filtered.subreddit) & (max_comments.max_rel_score == filtered.rel_score), how='inner').drop(max_comments.subreddit).drop(max_comments.max_rel_score)
    
    
    filtered = filtered.select('subreddit', 'author', 'rel_score')    
    filtered.coalesce(1).write.json(out_directory, mode='overwrite')



if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
