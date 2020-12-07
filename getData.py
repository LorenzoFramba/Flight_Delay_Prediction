#import findspark
#findspark.init()
import pyspark
import random

#from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *


class Data:
    def __init__(self, config):    
        self.cfg = config
        self.spark, self.sc = self._init_spark()
        self.df = self.getDataset(self.spark,self.sc)

    def _init_spark(self):
        spark = SparkSession.builder.appName("Project").getOrCreate()
        sc = spark.sparkContext
        return spark, sc

    def getDataset(self, spark, sc):
        sqlContext = SQLContext(sc)
        getDf = sqlContext.read.load(self.cfg.dataset, 
                      format='com.databricks.spark.csv', 
                      header='true',
                      delimiter=',',
                      inferSchema='true')
        return getDf

