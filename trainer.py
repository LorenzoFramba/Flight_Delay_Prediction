
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
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Bucketizer





class Trainer:

    def __init__(self, config, spark, sc, cleaned_data):     
        self.cfg = config
        self.spark =spark
        self.sc = sc
        self.df = cleaned_data.df
        self.X =cleaned_data.X
        self.oneHot = cleaned_data.oneHot
        self.varIdxer = cleaned_data.varIdxer
        self.bucketizer = cleaned_data.bucketizer


        self.train, self.test = self.df.randomSplit([.5, 0.1], seed=1234)  

        self.train = self.train.limit(1000000)
        self.test = self.test.limit(250000)


        #self.train, self.test = self.df.randomSplit([self.cfg.split_size_train / 100 , (100 - self.cfg.split_size_train ) / 100])

        
        
        #Views(config,df).correlation()
        
        results_LR=[]
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
                self.R2RF , self.maeRF, self.rmseRF = self.random_forest_train(features)
                features['R2RF'] = self.R2RF
                features['maeRF'] = self.maeRF
                features['rmseRF'] = self.rmseRF
                results_RF.append(features)
            
            for x in results_RF:
                self.Visualize_Results.append(x)
                print(x)

        elif(self.cfg.model == 'decision_tree_regression'):
            for features in self.X:
                self.R2DT , self.maeDT, self.rmseDT = self.decision_tree_regression_train(features)
                features['R2DT'] = self.R2DT
                features['maeDT'] = self.maeDT
                features['rmseDT'] = self.rmseDT
                results_DT.append(features)
            
            for x in results_DT:
                self.Visualize_Results.append(x)
                print(x)

        elif(self.cfg.model == 'all'):
            for features in self.X:
                
                self.R2LR , self.maeLR, self.rmseLR= self.linear_regression_train(features)
                self.R2DT , self.maeDT, self.rmseDT = self.decision_tree_regression_train(features)
                self.R2GBR , self.maeGBR, self.rmseGBR= self.gradient_boosted_tree_regression(features)
                self.R2RF , self.maeRF, self.rmseRF= self.random_forest_train(features)


                features['R2LR'] = self.R2LR
                features['maeLR'] = self.maeLR
                features['rmseLR'] = self.rmseLR

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
                    '\n Random Forest R2 : {R2RF}\t'
                    '\n Decision Tree Regression R2  : {R2DT}\t'
                    '\n Gradient Booster Tree Regression R2  : {R2GBR}\t'.format(
                    R2LR=self.R2LR, 
                    R2RF=self.R2RF, 
                    R2DT=self.R2DT, 
                    R2GBR = self.R2GBR)) 
        else:
            print("nothing was selected")


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

    def linear_regression_train(self,X):


        #A.append({ "name": "X1", "variables": ['DepDelay', 'TaxiOut']})
        #A.append({ "name": "X2", "variables": ['DepDelay', 'TaxiOut',  'HotDepTime']})
        #A.append({ "name": "X3", "variables": ['DepDelay', 'TaxiOut', 'HotIndOrigDest', 'HotDepTime']})
        #A.append({ "name": "X4", "variables": ['DepDelay', 'TaxiOut', 'HotDayOfWeek', 'HotMonth', 'Speed']})
        #A.append({ "name": "X5", "variables": ['DepDelay', 'TaxiOut', 'HotDayOfWeek', 'HotIndOrigDest', 'Speed']})
        #A.append({ "name": "X6", "variables": ['DepDelay', 'TaxiOut', 'HotIndOrigDest', 'Speed', 'HotCRSCatDepTime', 'HotCRSCatArrTime', 'HotDepTime']})

        #X[2]['variables']

        assembler = VectorAssembler(
                    inputCols=X['variables'],
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

        model = tvs.fit(self.train)
        predictions = model.transform(self.test)

        print("Linear Regression")
        print(X['variables'])

        R2, mae, rmse = self.metrics(predictions)

        return R2, mae, rmse


    def decision_tree_regression_train(self, X):
        
        assembler = VectorAssembler(inputCols=X['variables'], 
                                    outputCol='features')

        dt1 = DecisionTreeRegressor(featuresCol="features", 
                                    labelCol='ArrDelay')

        pipeline = Pipeline(stages=[self.bucketizer, self.varIdxer, self.oneHot, assembler, dt1])

        dtparamGrid = (ParamGridBuilder()
             .addGrid(dt1.maxDepth, [2, 5, 10, 20, 30])
             #.addGrid(dt.maxDepth, [2, 5, 10])
             .addGrid(dt1.maxBins, [10, 20, 40, 80, 100])
             #.addGrid(dt.maxBins, [10, 20])
             .build())

        dtcv = CrossValidator(estimator = pipeline,
                            estimatorParamMaps = dtparamGrid,
                            evaluator =RegressionEvaluator(labelCol="ArrDelay", 
                                                              metricName="rmse"),
                            numFolds = 3)

        model = dtcv.fit(self.train)
        predictions = model.transform(self.test)


        print("Decision Tree")
        print(X['variables'])
        R2, mae, rmse = self.metrics(predictions)

        return R2, mae, rmse

    def gradient_boosted_tree_regression(self, X):

        #(train, test) = gen_output.randomSplit([self.cfg.split_size_train / 100 , (100 - self.cfg.split_size_train ) / 100])

        assembler = VectorAssembler(inputCols=X['variables'], 
                                    outputCol='features')
        gbt = GBTRegressor(featuresCol="features", 
                           labelCol="ArrDelay", 
                           maxIter=10)
        pipeline = Pipeline(stages=[self.bucketizer, self.varIdxer, self.oneHot, assembler, gbt])

        #original
        TreeParamGrid = ParamGridBuilder()\
            .addGrid(gbt.maxDepth, [2, 10])\
            .addGrid(gbt.maxBins, [10, 20])\
            .build()

        tvs = CrossValidator(estimator=pipeline,
                                estimatorParamMaps=TreeParamGrid, #remove if don't want to use ParamGridBuilder
                                evaluator=RegressionEvaluator(labelCol="ArrDelay", 
                                                              metricName="rmse"),
                                numFolds=3)
                            #trainRatio=0.85)

        model = tvs.fit(self.train)
        predictions = model.transform(self.test)

        print("Gradient boosted tree")
        print(X['variables'])
        R2, mae, rmse = self.metrics(predictions)

        return R2, mae, rmse

    def random_forest_train(self, X):

        assembler = VectorAssembler(inputCols=X['variables'], 
                                    outputCol='features')
        
        rf = RandomForestRegressor(featuresCol="features", 
                                   labelCol='ArrDelay')

        pipeline = Pipeline(stages=[self.bucketizer, self.varIdxer, self.oneHot, assembler, rf])

        rfparamGrid = (ParamGridBuilder()
                    #.addGrid(rf.maxDepth, [2, 5, 10, 20, 30])
                    .addGrid(rf.maxDepth, [2, 5, 10])
                    #.addGrid(rf.maxBins, [10, 20, 40, 80, 100])
                    .addGrid(rf.maxBins, [5, 10, 20])
                    #.addGrid(rf.numTrees, [5, 20, 50, 100, 500])
                    .addGrid(rf.numTrees, [5, 20, 50])
                    .build())

        # Create 3-fold CrossValidator
        rfcv = CrossValidator(estimator = pipeline,
                            estimatorParamMaps = rfparamGrid,
                            evaluator = RegressionEvaluator(labelCol="ArrDelay", 
                                                              metricName="rmse"),
                            numFolds = 3)
        
        model = rfcv.fit(self.train)
        predictions = model.transform(self.test)

        print("Random Forest")
        print(X['variables'])
        R2, mae, rmse = self.metrics(predictions)

        return R2, mae, rmse
