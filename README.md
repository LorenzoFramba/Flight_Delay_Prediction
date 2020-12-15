# Airbus_Big_Data
A Big Data Assignment regarding Spark, with Airbus data fetched and linear regression model  

## Getting Started

### Dependencies

here are all the dependencies needed for the project 
* [python 3.5+](https://www.continuum.io/downloads)
* [pySpark](http://mirrors.viethosting.com/apache/spark/spark-2.4.7/spark-2.4.7-bin-hadoop2.7.tgz)
* [java8](https://www.oracle.com/java/technologies/java8.html)


- Here an easy script to download *pySpark* and *java8*. remember your path for the *installation_folder*
```bash

!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q http://mirrors.viethosting.com/apache/spark/spark-2.4.7/spark-2.4.7-bin-hadoop2.7.tgz
!tar xf spark-2.4.7-bin-hadoop2.7.tgz

```

```bash
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/installation_folder/spark-2.4.7-bin-hadoop2.7"

!pip3 install pyspark
```


- **Clone this repo**:
```bash
git clone https://github.com/LorenzoFramba/Airbus_Big_Data.git
cd Airbus_Big_Data
```


### To Start the program


- Select the *--path* at which the Airbus dataset is saved. If *--path* is not specified, the program assumes the Airbus is in the same folder as the project itselves. Make sure the name is 'year.csv' and *year* is a 4 digit number from 1987 to 2008. 

```bash
python main.py --dataset 'year.csv' 
```

- You also have the option to choose the train/test split (default is *75 / 25*), and the sample size for training with *--train_sample_size*. 

you also the ML model type  between *'linear_regression', 'generalized_linear_regression_train ', 'gradient_boosted_tree_regression',  'decision_tree_regression'* and *'random_forest'* (default : *linear_regression*). 
The *all* option will train and test all the models, compare their respective R2 and select the best performing one.

```bash
python main.py --dataset 'year.csv' --model 'linear_regression' --split_size_train 75 --train_sample_size 100000
```

- The selection of the variables is done by analyng patterns and correlation matrix ( select *--view* True to watch it). We selected this following variables together
"X1": ['DepDelay', 'TaxiOut']
"X2": ['DepDelay', 'TaxiOut',  'HotDepTime']     
"X3": ['DepDelay', 'TaxiOut', 'HotDayOfWeek', 'Speed']
"X4": ['DepDelay', 'TaxiOut', 'HotDayOfWeek', 'Speed', 'HotMonth']
"X5": ['DepDelay', 'TaxiOut', 'Speed', 'HotDepTime', 'HotCRSCatDepTime', 'HotCRSCatArrTime']

- By default, the model will run with the easier variable: X1. You have the option to use X5, which is the best performing one, by selecting "best" on *--variables*. You can also select "all" to try everything. 


```bash
python main.py --dataset 'year.csv' --model 'all' --split_size_train 75 --variables 'best' --view True 
```
