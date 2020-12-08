import pyspark
import random
from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *


class Data:
    def __init__(self, config):    
        self.cfg = config
        self.spark, self.sc = self._init_spark()
        self.checkValidity()
        self.df = self.getDataset(self.spark,self.sc)
        

    def _init_spark(self):
        spark = SparkSession.builder.appName("Project").getOrCreate()
        sc = spark.sparkContext
        return spark, sc

    def checkValidity(self):
        try:
            ('.csv') in self.cfg.dataset
        except ValueError:
            print("file not compatible")
            
        #try:
         #   print (self.cfg)
          #  raise print(error)
        #except InvalidArgError as e:
         #   print(e)

    def getDataset(self, spark, sc):
        sqlContext = SQLContext(sc)
        if self.cfg.path != "":
            self.cfg.path = self.cfg.path+'/'
            
        getDf = sqlContext.read.load(self.cfg.path+self.cfg.dataset, 
                      format='com.databricks.spark.csv', 
                      header='true',
                      delimiter=',',
                      inferSchema='true')
        return getDf

