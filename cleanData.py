import pyspark
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Bucketizer
from pyspark.sql.types import *
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql import functions as sf



class Clean: 
    def __init__(self, config, df, spark, sc):     
        self.cfg = config
        self.spark =spark
        self.sc = sc
        self.df = self.changeVar(df)
        self.df = self.removeNaN(self.df)
        #self.df= self.hotEncoding(self.df)
        self.X = self.variable_selection()
        

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
                       'Year', 
                       'TailNum', 
                       'CancellationCode' ] # Only those 3 I added up to delay, others 
                                                            # are delayed as is stated in the task
        df = df.drop(*col_to_drop)


        df = df.filter("Cancelled == 0") #select only those flights that happened
        df = df.drop("Cancelled")

        df = df.drop(*["UniqueCarrier", 
                       "DayofMonth", 
                       "FlightNum"]) #Droping unimportant categorical variables


        old_amount = df.count()
        df = df.na.drop("any")
        new_amount = df.count()
        
        print( "Before: " +str(old_amount) + ",\nAfter: " + str(new_amount) + ",\n%:"+str(round(new_amount/old_amount, 2)*100))

        df = df.withColumn('OrigDest', 
                    sf.concat(sf.col('Origin'),sf.lit('_'), sf.col('Dest')))
        df = df.drop(*["Origin", "Dest"])
        df = df.withColumn("Speed", sf.round(col("Distance") / col("CRSElapsedTime"), 2))
        
       
        return df

    def changeVar(self,df):

        # "ArrDelay" and "DepDelay" have string type. We cast them to Integer
        df = df.withColumn("ArrDelay", df["ArrDelay"].cast(IntegerType()))
        df = df.withColumn("DepDelay", df["DepDelay"].cast(IntegerType()))
        df = df.withColumn("CRSDepTime", df["CRSDepTime"].cast(IntegerType()))
        df = df.withColumn("CRSArrTime", df["CRSArrTime"].cast(IntegerType()))
        df = df.withColumn("DepTime", df["DepTime"].cast(IntegerType()))

        return df



    def variable_selection(self):

        X = []
        X.append({ "name": "X1", "variables": ['DepDelay', 'TaxiOut']})
        X.append({ "name": "X2", "variables": ['DepDelay', 'TaxiOut',  'HotDepTime']})
        X.append({ "name": "X3", "variables": ['DepDelay', 'TaxiOut', 'HotIndOrigDest', 'HotDepTime']})
        X.append({ "name": "X4", "variables": ['DepDelay', 'TaxiOut', 'HotDayOfWeek', 'HotMonth', 'Speed']})
        X.append({ "name": "X5", "variables": ['DepDelay', 'TaxiOut', 'HotDayOfWeek', 'HotIndOrigDest', 'Speed']})
        X.append({ "name": "X6", "variables": ['DepDelay', 'TaxiOut', 'HotIndOrigDest', 'Speed', 'HotCRSCatDepTime', 'HotCRSCatArrTime', 'HotDepTime']})

        return X


        