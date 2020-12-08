import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import isnan, when, count, col


class Clean: 
    def __init__(self, config, df, spark, sc):     
        self.cfg = config
        self.spark =spark
        self.sc = sc
        self.df = self.changeVar(df)
        self.df = self.removeNaN(self.df)
        #self.df = self.changeVar(self.df)

    def removeNaN(self, df):
        # removing as is stated in the task along with the 'Year' and 'DepTime'
        col_to_drop = ['ArrTime', 
                        'ActualElapsedTime', 
                        'AirTime', 
                        'TaxiIn', 
                        'Diverted', 
                        'CarrierDelay', 
                        'WeatherDelay', 
                        'NASDelay', 
                        'SecurityDelay', 
                        'LateAircraftDelay', 
                        'DepTime', #remove 
                        'Year']
        df = df.drop(*col_to_drop)

        # "CancelationCode" has too much "null" (98% of the data) we will remove it too. Others have no missing values except for "TailNum", that has only 127 values left.  
        df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()
        
        # deletion of the "CancelationCode" and droping rows that contain "TailNum"

        df = df.drop('TailNum')
        df = df.drop('CancellationCode')
        old_amount = df.count()
        df = df.na.drop("any")
        new_amount = df.count()
        
        print( "Before: " +str(old_amount) + ",\nAfter: " + str(new_amount) + ",\n%:"+str(round(new_amount/old_amount, 2)*100))
        #df =  df.filter(df.TailNum.isNotNull() )
       
        return df

    def changeVar(self,df):

        # "ArrDelay" and "DepDelay" have string type. We cast them to Integer
        df = df.withColumn("ArrDelay", df["ArrDelay"].cast(IntegerType()))
        df = df.withColumn("DepDelay", df["DepDelay"].cast(IntegerType()))

        return df

