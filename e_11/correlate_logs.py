import sys
assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
from pyspark.sql import SparkSession, Row
import re
from pyspark.sql import functions


# Regular expression to parse log lines
line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred.
    Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        return Row(hostname=m.group(1), bytes=int(m.group(2)))
    else:
        return None


def not_none(row):
    """
    Filter function to exclude None rows.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    # TODO: return an RDD of Row() objects
    rows = log_lines.map(line_to_row).filter(not_none)
    return rows


def main(in_directory):
    
    logs = spark.createDataFrame(create_row_rdd(in_directory))
     # TODO: calculate r.
    
    
    totals = logs.groupBy('hostname').agg(
        functions.count('*').alias('count'),
        functions.sum('bytes').alias('bytes')
    )

    sums = totals.select(functions.count('*').alias('n'),
        functions.sum('count').alias('sum_x'),
        functions.sum('bytes').alias('sum_y'),
        functions.sum(totals['count'] ** 2).alias('sum_x2'),
        functions.sum(totals['bytes'] ** 2).alias('sum_y2'),
        functions.sum(totals['count'] * totals['bytes']).alias('sum_xy')).first()

    n, sum_x, sum_y, sum_x2, sum_y2, sum_xy = sums

    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)) ** 0.5
    r = numerator / denominator

    print(f"r = {r}\nr^2 = {r**2}")
    #print('with built in function result:',totals.corr('count', 'bytes'))


if __name__ == '__main__':
    in_directory = sys.argv[1]
    spark = SparkSession.builder.appName('correlate logs').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory)
