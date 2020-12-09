import random
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




class Views:
    def __init__(self, config, df):    
        self.cfg = config
        self.df = df
        #self.correlation() 
        self.toPandas()

    def toPandas(self):
        #self.df_Pandas_25 = self.df.sample(False, 0.25, 42).toPandas()
        self.df_Pandas_25 = self.df.sample(False, 0.001, 42).toPandas()
        

    def correlation(self):
        corr_matrix = self.df.select([x[0] for x in self.df.dtypes if 'int' in x])
        corr_matrix.show(5)
        [(c[0], self.df.corr("ArrDelay", c[0])) for c in corr_matrix.dtypes]

    def correlation_matrix(self):
        fig, ax = plt.subplots(figsize=(11,9))         # Sample figsize in inches
        sns.heatmap(self.df_Pandas_25.corr(), annot = True, ax=ax,cbar_kws={"shrink": .5},  vmax = 1, vmin = -1, center = 0, cmap='coolwarm', linewidth =.5, linecolor ='black', square = True)
        #plt.show(block=False)


    def scatterPlot(self):
        features = ['DepDelay', 'TaxiOut']
        for feature in features:
            sns.regplot(x=self.df_Pandas_25["ArrDelay"], y=self.df_Pandas_25[feature])
            #plt.show(block=False)
            #plt.draw()
            #plt.pause(0.001)
            plt.show()

        



    

