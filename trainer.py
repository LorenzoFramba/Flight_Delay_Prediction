
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from  pyspark.sql.functions import abs


class Trainer:

    def __init__(self, config, df, spark, sc):     
        self.cfg = config
        self.spark =spark
        self.sc = sc
        self.df = df
        self.correlation() 

        if(self.cfg.model == 'linear_regression'):
            self.linear_regression_train()

        elif(self.cfg.model == 'al'):
            print("altro modello")
            #self.fuck
        else:
            print("nessuna selezione")


    def correlation(self):
        corr_matrix = self.df.select([x[0] for x in self.df.dtypes if 'int' in x])
        corr_matrix.show(5)
        [(c[0], self.df.corr("ArrDelay", c[0])) for c in corr_matrix.dtypes]


    def linear_regression_train(self):
        features = self.df.select(['DepDelay', 'TaxiOut'])
        assembler = VectorAssembler(
                    inputCols=features.columns,
                    outputCol="features")

        output = assembler.transform(self.df).select('features','ArrDelay')

        print(" train set ", self.cfg.split_size_train / 100)
        print(" test set ", (100 - self.cfg.split_size_train ) / 100 ) 
        train,test = output.randomSplit([self.cfg.split_size_train / 100 , (100 - self.cfg.split_size_train ) / 100])

        
        lin_reg = LinearRegression(featuresCol = 'features', 
                                   labelCol='ArrDelay',
                                   regParam=0.3,
                                   elasticNetParam=0.8 )

        linear_model = lin_reg.fit(train)


        print("Coefficients: " + str(linear_model.coefficients))
        print("\nIntercept: " + str(linear_model.intercept))

        trainSummary = linear_model.summary
        print("RMSE: %f" % trainSummary.rootMeanSquaredError)
        print("\nr2: %f" % trainSummary.r2)

        predictions = linear_model.transform(test)
        x =((predictions['ArrDelay']-predictions['prediction'])/predictions['ArrDelay'])*100
        predictions = predictions.withColumn('Accuracy',abs(x))
        predictions.select("prediction","ArrDelay","Accuracy","features").show(10)
        
        pred_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="ArrDelay",metricName="r2")
        print("R Squared (R2) on test data = %g" % pred_evaluator.evaluate(predictions))