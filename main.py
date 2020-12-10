from getData import Data
from cleanData import Clean
from trainer import Trainer
from visualization import Views

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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(config):

    findspark.init()

    data = Data(config)

    #df = Clean(config, data.df, data.spark, data.sc).df

    data_cleaned = Clean(config, data.df, data.spark, data.sc)

    Trainer(config,data_cleaned.df, data.spark, data.sc, data_cleaned.X)

    if(str(config.view).lower() == 'true'):
        Views(config,data_cleaned.df).correlation_matrix()
        Views(config,data_cleaned.df).scatterPlot()

    data.sc.stop()

if __name__ == '__main__':
    parser.add_argument('--dataset', type=str, default='2004.csv', required=True,   help='name of Airbus dataset to be used')
    parser.add_argument('--model', type=str, default='linear_regression', choices=['linear_regression', 'generalized_linear_regression_train', 'decision_tree_regression', 'gradient_boosted_tree_regression'  ,'random_forest',  'all'],   help='type of training model')
    parser.add_argument('--path', type=str, default='' )
    parser.add_argument('--split_size_train', type=int, default='75' , choices=range(1, 100),  help='percentage of observations in the training set')
    parser.add_argument('--regParam', type=float, default='0.3', help='specifies the regularization parameter in ALS, corresponds to λ' )
    parser.add_argument('--elasticNetParam', type=float, default='0.8' , help='elasticNetParam corresponds to α' ) 
    parser.add_argument("--view",  default=False, type=lambda x: (str(x).lower() == 'true'))

    config = parser.parse_args()
    print(config)
    main(config)

