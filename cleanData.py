
import pyspark
#from pyspark import SparkContext
#from pyspark.sql import SQLContext
#from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import isnan, when, count, col


class Clean: 
    def __init__(self, config, df, spark, sc):     
        self.cfg = config
        self.spark =spark
        self.sc = sc
        self.df = self.removeNaN(df)
        self.df = self.changeVar(self.df)

    def removeNaN(self, df):
        # removing as is stated in the task along with the 'Year'
        col_to_drop = ['ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 
                'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'Year']
        df = df.drop(*col_to_drop)
        df.show()

        # "CancelationCode" has too much "null" (98% of the data) we will remove it too. Others have no missing values except for "TailNum", that has only 127 values left.  
        df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()
        print(type(num))


        
        # deletion of the "CancelationCode" and droping rows that contain "TailNum"
        #df.na.drop("any").show(false) 
        df = df.drop('CancellationCode')
        df =  df.filter(df.TailNum.isNotNull() )
       
        return df

    def changeVar(self,df):

        # "ArrDelay" and "DepDelay" have string type. We cast them to Integer
        df = df.withColumn("ArrDelay", df["ArrDelay"].cast(IntegerType()))
        df = df.withColumn("DepDelay", df["DepDelay"].cast(IntegerType()))

        return df

