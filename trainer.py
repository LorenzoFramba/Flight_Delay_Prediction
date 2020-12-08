
from pyspark.sql.functions import abs
from pyspark.ml import Pipeline
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator





class Trainer:

    def __init__(self, config, df, spark, sc):     
        self.cfg = config
        self.spark =spark
        self.sc = sc
        self.df = df
        self.correlation() 

        if(self.cfg.model == 'linear_regression'):
            self.R2LR = self.linear_regression_train()

        elif(self.cfg.model == 'generalized_linear_regression_train'):
            self.R2GLR = self.generalized_linear_regression_train()

        elif(self.cfg.model == 'gradient_boosted_tree_regression'):
            self.R2GBR = self.Gradient_boosted_tree_regression()

        elif(self.cfg.model == 'random_forest'):
            train, test, featureIndexer = self.split_tree_forest()
            self.R2RF = self.random_forest_train(train, test, featureIndexer)

        elif(self.cfg.model == 'decision_tree_regression'):
            train, test, featureIndexer = self.split_tree_forest()
            self.R2DT = self.decision_tree_regression_train(train, test, featureIndexer)

        elif(self.cfg.model == 'all'):

            self.R2LR = self.linear_regression_train()
            self.R2GLR = self.generalized_linear_regression_train()
            train, test, featureIndexer = self.split_tree_forest()
            self.R2RF = self.random_forest_train(train, test, featureIndexer)
            self.R2DT = self.decision_tree_regression_train(train, test, featureIndexer)
            self.R2GBR = self.Gradient_boosted_tree_regression()

            print(  '\n Linear Regression R2 : {R2LR}\t'
                    '\n General Linear Regression R2 : {R2GLR}\t'
                    '\n Random Forest R2 : {R2RF}\t'
                    '\n Decision Tree Regression R2  : {R2DT}\t'
                    '\n Gradient Booster Tree Regression R2  : {R2GBR}\t'.format(
                    R2LR=self.R2LR, 
                    R2RF=self.R2RF, 
                    R2DT=self.R2DT, 
                    R2GBR = self.R2GBR,
                    R2GLR = self.R2GLR )) 
        else:
            print("nothing was selected")

    def correlation(self):
        corr_matrix = self.df.select([x[0] for x in self.df.dtypes if 'int' in x])
        corr_matrix.show(5)
        [(c[0], self.df.corr("ArrDelay", c[0])) for c in corr_matrix.dtypes]

    def split_tree_forest(self):
        features = self.df.select(['DepDelay', 
                              'TaxiOut', 
                              'ArrDelay'])

        gen_assembler = VectorAssembler(
            inputCols=features.columns[:-1],
            outputCol='features')

        gen_output = gen_assembler.transform(self.df).select('features',
                                                        'ArrDelay')

        featureIndexer = VectorIndexer(
                                        inputCol='features', 
                                        outputCol='IndexedFeatures').fit(gen_output)

        (train, test) = gen_output.randomSplit([self.cfg.split_size_train / 100 , (100 - self.cfg.split_size_train ) / 100])

        return train, test, featureIndexer

    def decision_tree_regression_train(self, train, test, featureIndexer):
        dt1 = DecisionTreeRegressor(
                                    featuresCol="IndexedFeatures", 
                                    labelCol='ArrDelay')

        pipeline = Pipeline(stages=[featureIndexer, dt1])

        # Train model.  This also runs the indexer.
        model = pipeline.fit(train)

        # Make predictions.
        predictions = model.transform(test)

        # Select example rows to display.
        predictions.select("prediction", 
                            'ArrDelay', 
                            "features").show(25)

        evaluator = RegressionEvaluator(
                            labelCol='ArrDelay', 
                            predictionCol="prediction", 
                            metricName="rmse")

        rmse = evaluator.evaluate(predictions)
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

        pred_evaluator = RegressionEvaluator(
                                            predictionCol="prediction", \
                                            labelCol="ArrDelay",
                                            metricName="r2")

        R2 = pred_evaluator.evaluate(predictions)
        print("R Squared (R2) on test data = %g" % R2)

        treeModel = model.stages[1]
        # summary only
        print(treeModel)
        return R2

    def Gradient_boosted_tree_regression(self):
        features = self.df.select(['DepDelay', 'TaxiOut', 'ArrDelay'])

        gen_assembler = VectorAssembler(
                                inputCols=features.columns[:-1],
                                outputCol='features')

        gen_output = gen_assembler.transform(self.df).select('features','ArrDelay')

        featureIndexer = VectorIndexer(
                                inputCol='features', 
                                outputCol='IndexedFeatures').fit(gen_output)
        (train, test) = gen_output.randomSplit([self.cfg.split_size_train / 100 , (100 - self.cfg.split_size_train ) / 100])

        gbt = GBTRegressor(featuresCol="IndexedFeatures", 
                           labelCol="ArrDelay", 
                           maxIter=10)

        pipeline = Pipeline(stages=[featureIndexer, gbt])
        model = pipeline.fit(train)

        predictions = model.transform(test)

        evaluator = RegressionEvaluator(
                            labelCol="ArrDelay", 
                            predictionCol="prediction", 
                            metricName="rmse")

        rmse = evaluator.evaluate(predictions)
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

        pred_evaluator = RegressionEvaluator(predictionCol="prediction", \
                                            labelCol="ArrDelay",
                                            metricName="r2")
        R2 = pred_evaluator.evaluate(predictions)
        print("R Squared (R2) on test data = %g" % R2)

        gbtModel = model.stages[1]
        print(gbtModel)  # summary only

        return R2



        
    def random_forest_train(self, train, test, featureIndexer):
        rf = RandomForestRegressor(
                                   featuresCol="IndexedFeatures", 
                                   labelCol='ArrDelay')

        pipeline = Pipeline(stages=[featureIndexer, rf])
        model = pipeline.fit(train)
        predictions = model.transform(test)
        predictions.select("prediction", "ArrDelay", "features").show(25)

        evaluator = RegressionEvaluator(
                                labelCol='ArrDelay', 
                                predictionCol="prediction", 
                                metricName="rmse")

        rmse = evaluator.evaluate(predictions)
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

        pred_evaluator = RegressionEvaluator(
                                predictionCol="prediction", \
                                labelCol="ArrDelay",
                                metricName="r2")

        R2 = pred_evaluator.evaluate(predictions)
        print("R Squared (R2) on test data = %g" % R2)

        rfModel = model.stages[1]
        print(rfModel)

        return R2

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
                                   regParam=self.cfg.regParam,
                                   elasticNetParam=self.cfg.elasticNetParam )

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
        r2 = pred_evaluator.evaluate(predictions)                 
        print("R Squared (R2) on test data = %g" % r2)

        return r2

    def generalized_linear_regression_train(self):
        features = self.df.select(['DepDelay', 'TaxiOut'])

        gen_assembler = VectorAssembler(
                        inputCols=features.columns,
                        outputCol='features')

        gen_output = gen_assembler.transform(self.df).select('features','ArrDelay')
        gen_train,gen_test = gen_output.randomSplit([self.cfg.split_size_train / 100 , (100 - self.cfg.split_size_train ) / 100])

        glr = GeneralizedLinearRegression(family="gaussian", 
                                          link="Identity", 
                                          maxIter=10, 
                                          regParam=self.cfg.regParam, 
                                          labelCol='ArrDelay')
        gen_model = glr.fit(gen_train)
        print("Coefficients: " + str(gen_model.coefficients))
        print("\nIntercept: " + str(gen_model.intercept)) 

        predictions = gen_model.transform(gen_test)
        x =((predictions['ArrDelay']-predictions['prediction'])/predictions['ArrDelay'])*100
        predictions = predictions.withColumn('Accuracy',abs(x))
        predictions.select("prediction","ArrDelay","Accuracy","features").show(10) 

        pred_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="ArrDelay",metricName="r2")

        R2 = pred_evaluator.evaluate(predictions)
        print("R Squared (R2) on test data = %g" % R2)

        return R2
                                        