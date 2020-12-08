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
```


- **Clone this repo**:
```bash
git clone https://github.com/LorenzoFramba/Airbus_Big_Data.git
cd Airbus_Big_Data
```

- Move your Airbus database in that folder. Make sure the name is 'year.csv' and *year* is a 4 digit number from 1987 to 2008. 


### To Start the program

```bash
python main.py --dataset 'year.csv' 
```

- You also have the option to choose the train/test split (default is *75 / 25*), and also the ML model type  between *'linear_regression', 'generalized_linear_regression_train ', 'decision_tree_regression'* and *'random_forest'* (default : *linear_regression*). The *all* option will train and test all the models, compare their respective R2 and select the best performing one.

- You have the option to set hyperparameters, such as *--elasticNetParam*or *--regParam* . 

```bash
python main.py --dataset 'year.csv' --model 'linear_regression' --split_size_train 75
```