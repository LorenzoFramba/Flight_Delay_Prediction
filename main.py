from getData import Data
from cleanData import Clean
from trainer import Trainer
from visualization import Views


import findspark
import argparse


parser = argparse.ArgumentParser()     


def main(config):
    """
    Captures the user input for the dataset and other customizations, process them and outputs the results

    """

    findspark.init()

    data = Data(config)

    if data.proceed:
        data_cleaned = Clean(config, data.df, data.spark, data.sc)
        trainer = Trainer(config, data.spark, data.sc, data_cleaned)

        if(str(config.view).lower() == 'true'):

            Views(config,data_cleaned.df,trainer.Visualize_Results).correlation_matrix()
            Views(config,data_cleaned.df,trainer.Visualize_Results).scatterPlot()

            if(config.model == 'all'):

                Views(config,data_cleaned.df,trainer.Visualize_Results).BarChart_R2()
                Views(config,data_cleaned.df,trainer.Visualize_Results).BarChart_MAE()
                Views(config,data_cleaned.df,trainer.Visualize_Results).BarChart_RMSE()

        data.sc.stop()

if __name__ == '__main__':
    parser.add_argument('--dataset', type=str, default='2004.csv', required=True,  help='name of Airbus dataset to be used')
    parser.add_argument('--model', type=str, default='linear_regression', choices=['linear_regression', 'decision_tree_regression', 'gradient_boosted_tree_regression'  ,'random_forest',  'all'],   help='type of training model')
    parser.add_argument('--variables', type=str, default='X1', choices=['X1', 'best', 'all'],   help='type of variables for training model')
    parser.add_argument('--path', type=str, default='' )
    parser.add_argument('--split_size_train', type=int, default='75' , choices=range(1, 100),  help='percentage of observations in the training set')
    parser.add_argument('--dataset_size', type=int, default='0',  help='Amount of samples for training/testing')
    parser.add_argument("--view",  default=False, type=lambda x: (str(x).lower() == 'true'), help='True for showing the correlation matrix and scatterplots' ) 

    config = parser.parse_args()
    print(config)
    main(config)

