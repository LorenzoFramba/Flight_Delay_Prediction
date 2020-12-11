
from visualization import Views
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

    def __init__(self, config, df, spark, sc, X):     
        self.cfg = config
        self.spark =spark
        self.sc = sc
        self.df = df
        self.X =X
        
        #Views(config,df).correlation()
        
        results_LR=[]
        results_GLR=[]
        results_RF=[]
        results_DT=[]
        results_GBR=[]
        results_all=[]

        self.Visualize_Results = []
        

        if(self.cfg.model == 'linear_regression'):

            for features in self.X:
                self.R2LR , self.maeLR, self.rmseLR = self.linear_regression_train(features)

                features['R2LR'] = self.R2LR
                features['maeLR'] = self.maeLR
                features['rmseLR'] = self.rmseLR
                results_LR.append(features)
        
            for x in results_LR:
                self.Visualize_Results.append(x)
                print(x)

        elif(self.cfg.model == 'gradient_boosted_tree_regression'):
            for features in self.X:
                self.R2GBR , self.maeGBR, self.rmseGBR = self.Gradient_boosted_tree_regression(features)
                features['R2GBR'] = self.R2GBR
                features['maeGBR'] = self.maeGBR
                features['rmseGBR'] = self.rmseGBR
                results_GBR.append(features)

            for x in results_GBR:
                self.Visualize_Results.append(x)
                print(x)

        elif(self.cfg.model == 'random_forest'):
            for features in self.X:
                train, test, featureIndexer = self.split_tree_forest(features)
                self.R2RF , self.maeRF, self.rmseRF = self.random_forest_train(train, test, featureIndexer)
                features['R2RF'] = self.R2RF
                features['maeRF'] = self.maeRF
                features['rmseRF'] = self.rmseRF
                results_RF.append(features)
            
            for x in results_RF:
                self.Visualize_Results.append(x)
                print(x)

        elif(self.cfg.model == 'decision_tree_regression'):
            for features in self.X:
                train, test, featureIndexer = self.split_tree_forest(features)
                self.R2DT , self.maeDT, self.rmseDT = self.decision_tree_regression_train(train, test, featureIndexer)
                features['R2DT'] = self.R2DT
                features['maeDT'] = self.maeDT
                features['rmseDT'] = self.rmseDT
                results_DT.append(features)
            
            for x in results_DT:
                self.Visualize_Results.append(x)
                print(x)

        elif(self.cfg.model == 'generalized_linear_regression_train'):
            for features in self.X:
                self.R2GLR, self.maeGLR, self.rmseGLR = self.generalized_linear_regression_train(features)

                features['R2GLR'] = self.R2GLR
                features['maeGLR'] = self.maeGLR
                features['rmseGLR'] = self.rmseGLR
                results_GLR.append(features)

            for x in results_GLR:
                self.Visualize_Results.append(x)
                print(x)

        elif(self.cfg.model == 'all'):
            for features in self.X:
                self.R2LR , self.maeLR, self.rmseLR= self.linear_regression_train(features)
                #self.R2GLR, self.maeGLR, self.rmseGLR  = self.generalized_linear_regression_train(features)
                train, test, featureIndexer = self.split_tree_forest(features)
                self.R2RF , self.maeRF, self.rmseRF= self.random_forest_train(train, test, featureIndexer)
                self.R2DT , self.maeDT, self.rmseDT = self.decision_tree_regression_train(train, test, featureIndexer)
                self.R2GBR , self.maeGBR, self.rmseGBR= self.Gradient_boosted_tree_regression(features)

                features['R2LR'] = self.R2LR
                features['maeLR'] = self.maeLR
                features['rmseLR'] = self.rmseLR

                #features['R2GLR'] = self.R2GLR
                #features['maeGLR'] = self.maeGLR
                #features['rmseGLR'] = self.rmseGLR

                features['R2RF'] = self.R2RF
                features['maeRF'] = self.maeRF
                features['rmseRF'] = self.rmseRF

                features['R2DT'] = self.R2DT
                features['maeDT'] = self.maeDT
                features['rmseDT'] = self.rmseDT

                features['R2GBR'] = self.R2GBR
                features['maeGBR'] = self.maeGBR
                features['rmseGBR'] = self.rmseGBR

                results_all.append(features)

            for x in results_all:
                self.Visualize_Results.append(x)
                print(x)

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

    def split_tree_forest(self, X):
        x = X['variables']+ ['ArrDelay']

        features = self.df.select(x)
        #features = self.df.select(['DepDelay', 
        #                      'TaxiOut', 
        #                      'ArrDelay'])

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

        pred_evaluator = RegressionEvaluator(
                                            predictionCol="prediction", \
                                            labelCol="ArrDelay",
                                            metricName="r2")
        mae_evaluator = RegressionEvaluator(labelCol='ArrDelay', 
                                            predictionCol="prediction", 
                                            metricName="mae")

        rmse = evaluator.evaluate(predictions)
        mae = mae_evaluator.evaluate(predictions)  
        R2 = pred_evaluator.evaluate(predictions)
        print("R Squared (R2) on test data = %g" % R2)
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

        treeModel = model.stages[1]
        # summary only
        print(treeModel)
        return R2, mae, rmse

    def Gradient_boosted_tree_regression(self, X):
        
        features = self.df.select(X['variables'])
        
        #features = self.df.select(['DepDelay', 'TaxiOut', 'ArrDelay'])

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


        mae_evaluator = RegressionEvaluator(labelCol='ArrDelay', 
                                            predictionCol="prediction", 
                                            metricName="mae")

        pred_evaluator = RegressionEvaluator(predictionCol="prediction", \
                                            labelCol="ArrDelay",
                                            metricName="r2")

        R2 = pred_evaluator.evaluate(predictions)
        mae = mae_evaluator.evaluate(predictions)
        rmse = evaluator.evaluate(predictions)

        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
        print("R Squared (R2) on test data = %g" % R2)

        gbtModel = model.stages[1]
        print(gbtModel)  # summary only

        return R2, mae, rmse



        
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

        mae_evaluator = RegressionEvaluator(labelCol='ArrDelay', 
                                            predictionCol="prediction", 
                                            metricName="mae")

        mae = mae_evaluator.evaluate(predictions)

        R2 = pred_evaluator.evaluate(predictions)
        print("R Squared (R2) on test data = %g" % R2)

        rfModel = model.stages[1]
        print(rfModel)

        return R2, mae, rmse

    def linear_regression_train(self,X):


        

        features = self.df.select(X['variables'])
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
        print("MAE: %f" % trainSummary.meanAbsoluteError)
        print("\nr2: %f" % trainSummary.r2)

        predictions = linear_model.transform(test)
        x =((predictions['ArrDelay']-predictions['prediction'])/predictions['ArrDelay'])*100
        predictions = predictions.withColumn('Accuracy',abs(x))
        predictions.select("prediction","ArrDelay","Accuracy","features").show(10)
        
        pred_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="ArrDelay",metricName="r2")
        mae_evaluator = RegressionEvaluator(labelCol='ArrDelay', 
                                            predictionCol="prediction", 
                                            metricName="mae")

        rmse_evaluator = RegressionEvaluator(labelCol='ArrDelay', 
                                            predictionCol="prediction", 
                                            metricName="rmse")

        mae = mae_evaluator.evaluate(predictions)
        rmse = rmse_evaluator.evaluate(predictions)
        r2 = pred_evaluator.evaluate(predictions)       

        print("MAE TEST: %f" % mae)
        print("rmse TEST: %f" % rmse)
        print("R Squared (R2) on test data = %g" % r2)

        return r2, mae, rmse

    def generalized_linear_regression_train(self,X):

        #features = self.df.select(X['variables'])
        features = self.df.select(['DepDelay', 'TaxiOut']) ## Currently, GeneralizedLinearRegression only supports number of features <= 4096. Found 4413 in the input dataset.


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

        mae_evaluator = RegressionEvaluator(labelCol='ArrDelay', 
                                            predictionCol="prediction", 
                                            metricName="mae")

        rmse_evaluator = RegressionEvaluator(labelCol='ArrDelay', 
                                            predictionCol="prediction", 
                                            metricName="rmse")

        mae = mae_evaluator.evaluate(predictions)
        rmse = rmse_evaluator.evaluate(predictions)      
        R2 = pred_evaluator.evaluate(predictions)


        print("R Squared (R2) on test data = %g" % R2)

        return R2, mae, rmse
                                        