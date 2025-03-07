import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('popular wikipedia pages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

schema = types.StructType([
    types.StructField('language', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('views', types.IntegerType()),
    types.StructField('bytes', types.LongType())
])

def extract_time(path):
    #pagecounts-YYYYMMDD-HHMMSS*
    file_name = path.split('/')[-1]
    return file_name[11:22]


path_to_time = functions.udf(extract_time, types.StringType())

def main(input_path, output_path):
    pageviews = spark.read.csv(input_path, schema=schema, sep=' ').withColumn('filename', functions.input_file_name())
    
    pageviews = pageviews.withColumn('time', path_to_time(pageviews['filename']))
    
    # no "Main_Page" "Special:"
    filtered = pageviews.filter(
        (pageviews['language'] == 'en') &
        (pageviews['title'] != 'Main_Page') &
        (~pageviews['title'].startswith('Special:'))).cache()
    
    # max views/time
    max_views_per_time = filtered.groupBy('time').agg(functions.max('views').alias('max_views'))

    filtered_alias = filtered.alias('f')
    max_views_per_time_alias = max_views_per_time.alias('m')
    popular_pages = filtered_alias.join(
        max_views_per_time_alias,
        (filtered_alias['time'] == max_views_per_time_alias['time']) &
        (filtered_alias['views'] == max_views_per_time_alias['max_views'])
    ).select(
filtered_alias['time'].alias('time'),     
        filtered_alias['title'].alias('title'),
        filtered_alias['views'].alias('views')   
    )

    
    popular_pages = popular_pages.orderBy('time', 'title')
    
    popular_pages.write.csv(output_path, mode='overwrite', header=True)


if __name__ == '__main__':
    input_path = sys.argv[1]  
    output_path = sys.argv[2]  
    main(input_path, output_path)
