from getData import Data
import findspark
import pandas as pd
import argparse
import altair as alt
import pyspark
import random

#from pyspark import SparkContext
#from pyspark.sql import SQLContext
#from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import isnan, when, count, col



parser = argparse.ArgumentParser()     

def main(config):

    findspark.init()
    df = Data(config).df

    # removing as is stated in the task along with the 'Year'
    col_to_drop = ['ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 
               'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'Year']
    df = df.drop(*col_to_drop)
    df.show()

    # "CancelationCode" has too much "null" (98% of the data) we will remove it too. Others have no missing values except for "TailNum", that has only 127 values left.  
    df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

    # deletion of the "CancelationCode" and droping rows that contain "TailNum"
    df = df.drop('CancellationCode')
    df =  df.filter(df.TailNum.isNotNull() )

    df.printSchema()

    # "ArrDelay" and "DepDelay" have string type. We cast them to Integer
    df = df.withColumn("ArrDelay", df["ArrDelay"].cast(IntegerType()))
    df = df.withColumn("DepDelay", df["DepDelay"].cast(IntegerType()))
    df.printSchema()


    corr_matrix = df.select([x[0] for x in df.dtypes if 'int' in x])

    # I guess it is too pythonic and we nees to change it's PEARSON CORRELATION

    [df.corr("ArrDelay", c[0]) for c in corr_matrix.dtypes]


    NON_corr_matrix = df.select([x[0] for x in df.dtypes if x[1] !='int']).show(5)

    df_Pandas_25 = df.sample(False, 0.25, 42).toPandas()




    alt.Chart(df_Pandas_25.sample(n=5000, random_state=1)).mark_point().encode(
        x='Origin',
        y='Distance',
        color='DayOfWeek',
    )


if __name__ == '__main__':
    parser.add_argument('--dataset', type=str, default='2004.csv')
    config = parser.parse_args()
    print(config)
    main(config)