# Airbus_Big_Data
A Big Data Assignment regarding Spark, with Airbus data fetched and linear regression model  

## Getting Started

### Dependencies

here are all the dependencies needed for the project 
* [python 3.5+](https://www.continuum.io/downloads)
* [pySpark](http://mirrors.viethosting.com/apache/spark/spark-2.4.7/spark-2.4.7-bin-hadoop2.7.tgz)
* [java8](https://www.oracle.com/java/technologies/java8.html)


- You can also run this script your chosen *installation_folder*
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

- Move your Airbus database in that folder. Make sure the name is 'year.csv'


### To Start the program

```bash
python main.py --dataset 'year.csv'
```
