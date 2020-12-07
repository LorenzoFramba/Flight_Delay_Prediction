from getData import Data
from cleanData import Clean
from trainer import Trainer
import findspark
import pandas as pd
import argparse
import altair as alt
import pyspark
import random

from pyspark.sql.types import *
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.feature import VectorAssembler



parser = argparse.ArgumentParser()     

def main(config):

    findspark.init()

    data = Data(config)

    df = Clean(config, data.df, data.spark, data.sc).df

    Trainer(config,df, data.spark, data.sc)
    #df.printSchema()
    #df_Pandas_25 = df.sample(False, 0.25, 42).toPandas()

    #alt.Chart(df_Pandas_25.sample(n=5000, random_state=1)).mark_point().encode(
    #    x='Origin',
    #    y='Distance',
    #    color='DayOfWeek',
    #)


if __name__ == '__main__':
    parser.add_argument('--dataset', type=str, default='2004.csv')
    parser.add_argument('--model', type=str, default='linear_regression')
    parser.add_argument('--split_size_train', type=int, default='75')
    config = parser.parse_args()
    print(config)
    main(config)