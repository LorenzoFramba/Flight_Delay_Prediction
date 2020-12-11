
from visualization import Views
from pyspark.sql.functions import abs
from pyspark.ml import Pipeline
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator





class Trainer:

    def __init__(self, config, spark, sc, cleaned_data):     
        self.cfg = config
        self.spark =spark
        self.sc = sc
        self.df = cleaned_data.df
        self.X =cleaned_data.X
        self.bucketizer = cleaned_data.bucketizer
        self.varIdxer = cleaned_data.varIdxer
        self.oneHot = cleaned_data.oneHot


        
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
                self.R2GBR , self.maeGBR, self.rmseGBR = self.gradient_boosted_tree_regression(features)
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
                self.R2GBR , self.maeGBR, self.rmseGBR= self.gradient_boosted_tree_regression(features)

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


    def linear_regression_train(self,X):



        train, test = self.df.randomSplit([.5, 0.1], seed=1234)  

        train = train.limit(1000000)
        test = test.limit(250000)

        features = self.df.select(X['variables'])
        assembler = VectorAssembler(
                    inputCols=features.columns,
                    outputCol="features")

        lin_reg = LinearRegression(featuresCol = 'features', labelCol="ArrDelay")


        pipeline = Pipeline(stages=[self.bucketizer, self.varIdxer, self.oneHot, assembler,  lin_reg])


        linParamGrid = ParamGridBuilder()\
                        .addGrid(lin_reg.regParam, [0.1, 0.01]) \
                        .addGrid(lin_reg.fitIntercept, [False, True])\
                        .addGrid(lin_reg.elasticNetParam, [0.0, 1.0])\
                        .build()


        tvs = CrossValidator(estimator=pipeline,\
                           estimatorParamMaps = linParamGrid,  
                           evaluator=RegressionEvaluator(labelCol="ArrDelay", metricName="rmse"),\
                           numFolds=3)
                           #trainRatio=0.85)

        model = tvs.fit(train)

        predictions = model.transform(test)

        R2, mae, rmse = self.metrics(predictions)


        return R2, mae, rmse

    def split_tree_forest(self, X):
        x = X['variables']+ ['ArrDelay']

        features = self.df.select(x)
        
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
        R2, mae, rmse = self.metrics(predictions)

        treeModel = model.stages[1]
        # summary only
        print(treeModel)
        return R2, mae, rmse

    def gradient_boosted_tree_regression(self, X):


        #(train, test) = gen_output.randomSplit([self.cfg.split_size_train / 100 , (100 - self.cfg.split_size_train ) / 100])

        train, test = self.df.randomSplit([.5, 0.1], seed=1234)  

        train = train.limit(1000000)
        test = test.limit(250000)
        
        features = self.df.select(X['variables'])
        
        #features = self.df.select(['DepDelay', 'TaxiOut', 'ArrDelay'])

        assembler = VectorAssembler(inputCols=features, outputCol='features')

        gbt = GBTRegressor(featuresCol="features", labelCol="ArrDelay", maxIter=10)

        pipeline = Pipeline(stages=[self.bucketizer, self.varIdxer, self.oneHot, assembler, gbt])

        TreeParamGrid = ParamGridBuilder()\
            .addGrid(gbt.maxDepth, [2, 10])\
            .addGrid(gbt.maxBins, [10, 20])\
            .build()

        tvs = CrossValidator(estimator=pipeline,
                                estimatorParamMaps=TreeParamGrid, #remove if don't want to use ParamGridBuilder
                                evaluator=RegressionEvaluator(labelCol="ArrDelay", metricName="rmse"),
                                numFolds=3)
                            #trainRatio=0.85)

        model = tvs.fit(train)

        predictions = model.transform(test)

        R2, mae, rmse = self.metrics(predictions)

        gbtModel = model.stages[1]
        print(gbtModel)  # summary only

        return R2, mae, rmse


    def metrics(self, predictions):


        x =((predictions['ArrDelay']-predictions['prediction'])/predictions['ArrDelay'])*100
        predictions = predictions.withColumn('Accuracy',abs(x))

        rmse_evaluator = RegressionEvaluator(
                            labelCol="ArrDelay", 
                            predictionCol="prediction", 
                            metricName="rmse")


        mae_evaluator = RegressionEvaluator(labelCol='ArrDelay', 
                                            predictionCol="prediction", 
                                            metricName="mae")

        R2_evaluator = RegressionEvaluator(predictionCol="prediction", \
                                            labelCol="ArrDelay",
                                            metricName="r2")

        R2 = R2_evaluator.evaluate(predictions)
        mae = mae_evaluator.evaluate(predictions)
        rmse = rmse_evaluator.evaluate(predictions)

        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
        print("Mean Absolute Error (MAE) on test data = %g" % mae)
        print("R Squared (R2) on test data = %g" % R2)

        return R2, mae, rmse

        

    def random_forest_train(self, train, test, featureIndexer):
        rf = RandomForestRegressor(
                                   featuresCol="IndexedFeatures", 
                                   labelCol='ArrDelay')

        pipeline = Pipeline(stages=[featureIndexer, rf])
        model = pipeline.fit(train)
        predictions = model.transform(test)


        R2, mae, rmse = self.metrics(predictions)

        rfModel = model.stages[1]
        print(rfModel)

        return R2, mae, rmse


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


        R2, mae, rmse = self.metrics(predictions)

        return R2, mae, rmse
                                        