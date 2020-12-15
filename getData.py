import pyspark
import random
from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *


class Data:
    def __init__(self, config):    
        self.cfg = config
        self.spark, self.sc = self._init_spark()
        self.checkFormatValidity()

    def _init_spark(self):
        spark = SparkSession.builder.appName("Project").getOrCreate()
        sc = spark.sparkContext
        return spark, sc

    def checkFormatValidity(self):
        try:
            if ('.csv') in self.cfg.dataset:
                self.proceed = True
                self.df = self.getDataset(self.spark,self.sc)
            else:
                self.proceed = False
                print('File format not correct. It is required to provide a .csv file')
        except ValueError:
            print("file not compatible")

            

    def getDataset(self, spark, sc):
        sqlContext = SQLContext(sc)
        if self.cfg.path != "":
            self.cfg.path = self.cfg.path+'/'

        getDf = sqlContext.read.load(self.cfg.path+self.cfg.dataset, 
                      format='com.databricks.spark.csv', 
                      header='true',
                      delimiter=',',
                      inferSchema='true')

        max = getDf.count()

        if(self.cfg.dataset_size > max):
            self.proceed = False
            print("Please, select a dataset_size smaller than ", max)

        return getDf

            

