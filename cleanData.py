from pyspark.ml.feature import StringIndexer, OneHotEncoder, Bucketizer
from pyspark.sql.types import *
from pyspark.sql.functions import col
from pyspark.sql import functions as sf


class Clean:
    """
    Conducts data preprocessing and transformation of the selected feature vectors
    """

    def __init__(self, config, df, spark, sc):
        self.cfg = config
        self.spark = spark
        self.sc = sc
        self.df = self.changeVar(df)
        self.df = self.filterAndTransform(self.df)
        self.X = self.variable_selection()
        self.OneHotEncoder()

    def filterAndTransform(self, df):
        """
        Removes rows containing Nan, filters out cancelled flights and creates new variables as
        the combination of the existing.

        """

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
                       'CancellationCode']  # Only those 3 I added up to delay, others
        # are delayed as is stated in the task
        df = df.drop(*col_to_drop)

        df = df.filter("Cancelled == 0")  # select only those flights that happened
        df = df.drop("Cancelled")

        df = df.drop(*["UniqueCarrier",
                       "DayofMonth",
                       "FlightNum"])  # Droping unimportant categorical variables

        df = df.na.drop("any")

        df = df.withColumn('OrigDest',
                           sf.concat(sf.col('Origin'), sf.lit('_'), sf.col('Dest')))
        df = df.drop(*["Origin", "Dest"])
        df = df.withColumn("Speed", sf.round(col("Distance") / col("CRSElapsedTime"), 2).cast(DoubleType()))

        return df

    def changeVar(self, df):
        """
        Ensures that variables are assigned to the right data type

        """

        # "ArrDelay" and "DepDelay" have string type. We cast them to Integer
        df = df.withColumn("ArrDelay", df["ArrDelay"].cast(IntegerType()))
        df = df.withColumn("DepDelay", df["DepDelay"].cast(IntegerType()))
        df = df.withColumn("CRSDepTime", df["CRSDepTime"].cast(IntegerType()))
        df = df.withColumn("CRSArrTime", df["CRSArrTime"].cast(IntegerType()))
        df = df.withColumn("DepTime", df["DepTime"].cast(IntegerType()))
        df = df.withColumn("DayOfWeek", df["DayOfWeek"].cast(IntegerType()))

        return df

    def OneHotEncoder(self):
        """
        Converts string-type categories to indexes, splits continuous data interval to indexes,
        encodes the categorical data using One-Hot encoding.

        """
        splits = [-float("inf"), 500, 1200, 1700, float("inf")]
        self.bucketizer = Bucketizer(splitsArray=[splits, splits, splits],
                                     inputCols=["CRSDepTime",
                                                "CRSArrTime",
                                                "DepTime"],
                                     outputCols=["CatCRSDepTime",
                                                 "CatCRSArrTime",
                                                 "CatDepTime"])

        self.varIdxer = StringIndexer(inputCol="OrigDest",
                                      outputCol="IndOrigDest").setHandleInvalid("skip")

        self.oneHot = OneHotEncoder(inputCols=['Month',
                                               'DayOfWeek',
                                               'CatCRSDepTime',
                                               'CatCRSArrTime',
                                               'IndOrigDest',
                                               'CatDepTime'],
                                    outputCols=['HotMonth',
                                                'HotDayOfWeek',
                                                'HotCRSCatDepTime',
                                                'HotCRSCatArrTime',
                                                'HotIndOrigDest',
                                                'HotDepTime']).setHandleInvalid("keep")

    def variable_selection(self):
        """
        Based on user input selects the variables vectors to process

        """
        X = []

        if self.cfg.variables == 'X1':
            X.append({"name": "X1", "variables": ['DepDelay', 'TaxiOut']})
        elif self.cfg.variables == 'all':
            X.append({"name": "X1", "variables": ['DepDelay', 'TaxiOut']})
            X.append({"name": "X2", "variables": ['DepDelay', 'TaxiOut', 'HotDepTime']})
            X.append({"name": "X3", "variables": ['DepDelay', 'TaxiOut', 'HotDayOfWeek', 'Speed']})
            X.append({"name": "X4", "variables": ['DepDelay', 'TaxiOut', 'HotDayOfWeek', 'Speed', 'HotMonth']})
            X.append({"name": "X5", "variables": ['DepDelay', 'TaxiOut', 'Speed', 'HotDepTime', 'HotCRSCatArrTime']})
        elif self.cfg.variables == 'best':
            X.append({"name": "X5", "variables": ['DepDelay', 'TaxiOut', 'Speed', 'HotDepTime', 'HotCRSCatArrTime']})
        return X
