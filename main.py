from getData import Data
from cleanData import Clean
from trainer import Trainer
from pyspark.sql.types import *
from pyspark.sql.functions import isnan, when, count, col
from pyspark.ml.feature import VectorAssembler
import findspark
import pandas as pd
import argparse
import altair as alt
import pyspark
import random


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

    data.sc.stop()

if __name__ == '__main__':
    parser.add_argument('--dataset', type=str, default='2004.csv', required=True,   help='name of Airbus dataset to be used')
    parser.add_argument('--model', type=str, default='linear_regression', choices=['linear_regression', 'generalized_linear_regression_train', 'decision_tree_regression', 'random_forest',  'all'],   help='type of training model')
    parser.add_argument('--split_size_train', type=int, default='75' , choices=range(1, 100),  help='percentage of observations in the training set')
    config = parser.parse_args()
    print(config)
    main(config)