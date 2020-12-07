from getData import Data
from cleanData import Clean
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
from pyspark.ml.feature import VectorAssembler



parser = argparse.ArgumentParser()     

def main(config):

    findspark.init()

    data = Data(config)
    spark = data.spark
    sc = data.sc

    df = Clean(config, data.df, spark, sc).df
    #df.printSchema()


    corr_matrix = df.select([x[0] for x in df.dtypes if 'int' in x])

    # I guess it is too pythonic and we nees to change it's PEARSON CORRELATION

    [df.corr("ArrDelay", c[0]) for c in corr_matrix.dtypes]


    NON_corr_matrix = df.select([x[0] for x in df.dtypes if x[1] !='int']).show(5)



    df_Pandas_25 = df.sample(False, 0.25, 42).toPandas()

    #alt.Chart(df_Pandas_25.sample(n=5000, random_state=1)).mark_point().encode(
    #    x='Origin',
    #    y='Distance',
    #    color='DayOfWeek',
    #)


if __name__ == '__main__':
    parser.add_argument('--dataset', type=str, default='2004.csv')
    parser.add_argument('--model', type=str, default='Linear_Regression')
    config = parser.parse_args()
    print(config)
    main(config)