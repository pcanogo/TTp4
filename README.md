# TTp4
Proyecto  4 de Topicos Especiales de Telematica 
Pablo Cano

Spark text clustering with kmeans

## Dependencies 
This proyect requires a Spark and Hadoop File Distribution System environment to work.

## Text Selection
Choose text collection
```
documents = sc.wholeTextFiles("hdfs:///distributed/file/directory")
```

Type in the files to be collected to documents variable.


## How to run
To run enter the following command:

To run on client 
```
$ spark-submit --master yarn --deploy-mode client sKmeans.py 
```

To run on cluster 
```
$ spark-submit --master yarn --deploy-mode cluster sKmeans.py 
```

*Note:* If a lot of resources a required, run on cluster.
