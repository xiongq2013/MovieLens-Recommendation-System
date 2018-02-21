# Databricks notebook source
""""
## Movie recommender system with Spark machine learning
### MovieLens
MovieLens is a project developed by GroupLens, a research laboratory at the University of Minnesota. 
MovieLens provides an online movie recommender application that uses anonymously-collected data to improve recommender algorithms. 
Anyone can try the app for free and get movies recommendations. To help people develop the best recommendation algorithms, 
MovieLens also released several data sets. In this notebook, we'll use the latest data set, which has two sizes.

The full data set consists of more than 24 million ratings across more than 40,000 movies by more than 250,000 users. 
The file size is kept under 1GB by using indexes instead of full string names.

The small data set is a subset of the full data set. It's generally a good idea to start building a working program with 
a small data set to get faster performance while interacting, exploring, and getting errors with your data. When we have 
a fully working program, we can apply the same code to the larger data set, possibly on a larger cluster of processors. 
We can also minimize memory consumption by limiting the data volume as much as possible, for example, by using indexes.
# MAGIC 
# MAGIC ### Spark machine learning library
# MAGIC The library has two packages:
# MAGIC * spark.mllib contains the original API that handles data in RDDs. It's in maintenance mode, but fully supported.
# MAGIC * spark.ml contains a newer API for constructing ML pipelines. It handles data in DataFrames. It's being actively enhanced.

# COMMAND ----------

# MAGIC %md ### 1. Load the data
# MAGIC We'll create Spark DataFrames, which are similar to R or pandas DataFrames, but can be distributed on a cluster of Spark executors, which can potentially scale up to thousands of machines. DataFrames are one of the easiest and best performing ways of manipulating data with Spark, but they require structured data in formats or sources such as CSV, Parquet, JSON, or JDBC.
# MAGIC 
# MAGIC Here we will use files movie.csv and ratings.csv.

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/movies.csv

# COMMAND ----------

movies = spark.read.format("csv")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load('/FileStore/tables/movies.csv')

movies.cache()
display(movies)

# COMMAND ----------

ratings = spark.read.format("csv")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load('/FileStore/tables/ratings.csv')

ratings.cache()
display(ratings)

# COMMAND ----------

# MAGIC %md ### 2. Explore the data with Spark APIs

# COMMAND ----------

# Print out the schema of the two tables
movies.printSchema()
ratings.printSchema()

# COMMAND ----------

# MAGIC %md Run the describe( ) method to see the count, mean, standard deviation, minimum, and maximum values for the data in each column.

# COMMAND ----------

display(ratings.describe())

# COMMAND ----------

# See how many distinct ratings, users and movies we have total. Also the number of good movies (with at least one rating >4).
numRatings=ratings.count()
numUsers=ratings.select("userId").distinct().count()
numMovies=ratings.select("movieId").distinct().count()

print "Number of different ratings: "+str(numRatings)
print "Number of different users: "+str(numUsers)
print "Number of different movies: "+str(numMovies)

numGoodMovies=ratings.filter("rating > 4").select("movieId").distinct().count()
print "Number of movies with at least one rating strictly higher than 4: "+str(numGoodMovies)

# COMMAND ----------

# Counts the ratings for each movie
movies_counts=ratings.select("movieId","rating").groupBy("movieId").count().orderBy("movieId").toDF("movieId","counts")
display(movies_counts)

# COMMAND ----------

# MAGIC %md ### 3. Visualize the data
# MAGIC 
# MAGIC we'll use the Seaborn and matplotlib matplotlib libraries to create graphs. The Seaborn library works with the matplotlib library to graph statistical data.

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
ratings_DF=ratings.toPandas()

# COMMAND ----------

# MAGIC %md Create the graph on a larger scale with the color palette:

# COMMAND ----------

sns.lmplot(x="userId",y="movieId",hue="rating",data=ratings_DF,fit_reg=False,size=10,aspect=2,palette=sns.diverging_palette(10,133,sep=80,n=10))
display(plt.show())

# COMMAND ----------

# MAGIC %md On this matrix, you'll notice gaps in the data: some movies and users are missing. This is because you're using a subset of the data.
# MAGIC 
# MAGIC Nevertheless, you can identify some patterns. Some users always give positive reviews of movies. Some movies are rated a lot, which could be for different reasons, such as the first release of the MovieLens website had a much smaller catalog, or the movies are more famous.

# COMMAND ----------

# MAGIC %md ### 4. Build the recommender system
# MAGIC 
# MAGIC There are different methods for building a recommender system, such as, user-based, content-based, or collaborative filtering. Collaborative filtering calculates recommendations based on similarities between users and products. For example, collaborative filtering assumes that users who give the similar ratings on the same movies will also have similar opinions on movies that they haven't seen.
# MAGIC 
# MAGIC The alternating least squares (ALS) algorithm provides collaborative filtering between users and products to find products that the customers might like, based on their previous ratings.
# MAGIC 
# MAGIC In this case, the ALS algorithm will create a matrix of all users (row) versus all movies (col). Most cells in the matrix will be empty. An empty cell means the user hasn't reviewed the movie yet. The ALS algorithm will fill in the probable ratings, based on similarities between user ratings and similarities between movies. The algorithm uses the least squares computation to minimize the estimation errors, and alternates between solving for movie factors and solving for user factors.
# MAGIC 
# MAGIC The following trivial example gives you an idea of the problem to solve. However, keep in mind that the general problem is much harder because the matrix often has far more missing values.

# COMMAND ----------

# create table so we can use SQL
ratings.createOrReplaceTempView("ratings_sql")

# COMMAND ----------

# MAGIC %sql  /* Check the size of the ratings matrix   */
# MAGIC select *, nb_ratings/matrix_size*100 as percentage
# MAGIC from (
# MAGIC   select *, nb_users*nb_movies as matrix_size
# MAGIC   from (
# MAGIC     select count(distinct(userId)) as nb_users, count(distinct(movieId)) as nb_movies, count(*) as nb_ratings
# MAGIC     from ratings_sql
# MAGIC   )
# MAGIC   )

# COMMAND ----------

# MAGIC %md Less than 2% of the matrix is filled.

# COMMAND ----------

# MAGIC %md ####4.1 Train the model
# MAGIC 
# MAGIC Use the SparkML ALS algorithm to train a model to provide recommendations. The mandatory parameters to the ALS algorithm are the columns that identify the users, the items, and the ratings. Run the fit() method to train the model:

# COMMAND ----------

from pyspark.ml.recommendation import ALS
model= ALS(userCol="userId",itemCol="movieId",ratingCol="rating").fit(ratings)

# COMMAND ----------

# MAGIC %md #### 4.2 Run the model
# MAGIC 
# MAGIC Run the transform( ) method to score the model and output a DataFrame with an additional prediction column that shows the predicted rating

# COMMAND ----------

predictions=model.transform(ratings)
display(predictions)

# COMMAND ----------

# MAGIC %md You can see that many of the predictions are close to the actual ratings.

# COMMAND ----------

# MAGIC %md #### 4.3 Evaluate the model
# MAGIC 
# MAGIC After you apply a model to a data set, you should evaluate the performance of the model by comparing the predicted values with the original values. Use the RegressionEvaluator method to compare continuous values with the root mean squared calculation. The root mean squared error calculation measures the average of the squares of the errors between what is estimated and the existing data. The lower the mean squared error value, the more accurate the model.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
print "the root mean squared error is:"+str(evaluator.evaluate(predictions))

# COMMAND ----------

# MAGIC %md You want the performance score to improve with your design iterations so that the model is improved. But notice that you just ran the training and the scoring on the same data set. That's something that you won't normally do because you usually want to predict values that you don't already know! Therefore, this result is nonsense. To accurately evaluate the model, it's common practice in machine learning to split the data set between a training data set to train the model, and a test data set to compare the predicted results with the original results. This process is called cross-validation. Not doing cross-validation often leads to overfitting, which occurs when the model is too specific to the training data set and does not perform well on a more general data set.

# COMMAND ----------

# MAGIC %md #### 4.4 Split the data set
# MAGIC 
# MAGIC Split your ratings data set between an 80% training data set and a 20% test data set. Then rerun the steps to train, run, and evaluate the model.

# COMMAND ----------

ratings_train, ratings_test=ratings.randomSplit([0.8, 0.2])
model= ALS(userCol="userId",itemCol="movieId",ratingCol="rating").fit(ratings_train)
predictions=model.transform(ratings_test)
display(predictions)

# COMMAND ----------

evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
print "the root mean squared error is:"+str(evaluator.evaluate(predictions))

# COMMAND ----------

# MAGIC %md You might get the value nan (not a number) from the previous cell.

# COMMAND ----------

# MAGIC %md #### 4.5 Handle NaN results
# MAGIC A NaN result is because the model can't predict values for users for which there's no data. A temporary workaround is to exclude rows with predicted NaN values or to replace them with a constant, for example, the general mean rating. However, to map to a real business problem, the data scientist, in collaboration with the business owner, must define what happens if such an event occurs. For example, you can provide no recommendation for a user until that user rates a few items. Alternatively, before user rates five items, you can use a user-based recommender system that's based on the user's profile (that's another recommender system to develop).
# MAGIC 
# MAGIC Replace predicted NaN values with the average rating and evaluate the model:

# COMMAND ----------

ratings_avg=ratings.select('rating').groupBy().avg().first()[0]
evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
print "the root mean squared error is:"+str(evaluator.evaluate(predictions.na.fill(ratings_avg)))

# COMMAND ----------

# MAGIC %md Or exclude predicted NaN values and evaluate the model:

# COMMAND ----------

evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
print "the root mean squared error is:"+str(evaluator.evaluate(predictions.na.drop()))

# COMMAND ----------

# MAGIC %md Obviously, you get lower performance than with the previous model, but you're protected against overfitting: you will actually get this level of performance on new incoming data!

# COMMAND ----------

# MAGIC %md #### 4.6 Improve the performance score
# MAGIC If you run the randomSplit(), fit(), transform(), and evaluate() functions several times, you won't always get the same performance score. This is because the randomSplit() and ALS() functions have some randomness. To get a more precise performance score, run the model several times and then compute the average performance score. This process is really close to k-fold cross validation.
# MAGIC 
# MAGIC Create a repeatALS function that trains, runs, and evaluates the model multiple times:

# COMMAND ----------

import numpy as np
def repeatALS (data, k, userCol="userId", itemCol="movieId",ratingCol="rating",metricName="rmse"):
  evaluations=[]
  for i in range(1,k+1):
    train,test=data.randomSplit([k-1.0,1.0])
    model= ALS(userCol=userCol,itemCol=itemCol,ratingCol=ratingCol).fit(train)
    predictions=model.transform(test)
    evaluator=RegressionEvaluator(metricName=metricName,labelCol="rating",predictionCol="prediction")
    evaluation=evaluator.evaluate(predictions.na.drop())
    print "Loop"+str(i)+":"+metricName+"="+str(evaluation)
    evaluations.append(evaluation)
  return np.mean(evaluations)

# COMMAND ----------

repeatALS(ratings,k=4)

# COMMAND ----------

# MAGIC %md The computed performance score is more stable this way.
# MAGIC 
# MAGIC Create a kfoldALS function that also trains, runs, and evaluates the model, but splits up the data between training and testing data sets in a different way. The original data set is split into k data sets. Each of the k iterations of the function uses a different data set for testing and the other data sets for training.

# COMMAND ----------

def kfoldALS (data, k, userCol="userId", itemCol="movieId",ratingCol="rating",metricName="rmse"):
  evaluations=[]
  weights=[1.0]*k
  splits=data.randomSplit(weights)
  for i in range(0,k):
    test=splits[i]
    train=spark.createDataFrame(sc.emptyRDD(),data.schema)
    for j in range(0,k):
      if i==j: continue
      else: train=train.union(splits[j])
    model= ALS(userCol=userCol,itemCol=itemCol,ratingCol=ratingCol).fit(train)
    predictions=model.transform(test)
    evaluator=RegressionEvaluator(metricName=metricName,labelCol="rating",predictionCol="prediction")
    evaluation=evaluator.evaluate(predictions.na.drop())
    print "Loop"+str(i)+":"+metricName+"="+str(evaluation)
    evaluations.append(evaluation)
  return np.mean(evaluations)

# COMMAND ----------

kfoldALS(ratings,k=4)

# COMMAND ----------

kfoldALS(ratings,k=10)

# COMMAND ----------

# MAGIC %md The bigger the training set is, the better performances you get. A general assumption in machine learning is that more data usually beats a better algorithm. You can easily improve this performance score by using the full data set.

# COMMAND ----------

# MAGIC %md #### 4.7 Improve the model
# MAGIC So now, how can we improve this model? Machine learning algorithms have hyperparameters that control how the algorithm works.
# MAGIC 
# MAGIC The ALS algorithm has this signature. The ALS hyperparameters are:
# MAGIC 
# MAGIC * rank = the number of latent factors in the model
# MAGIC * maxIter = the maximum number of iterations
# MAGIC * regParam = the regularization parameter
# MAGIC To test several values for those hyperparameters and choose the best configuration, it's common practice to define a grid of parameter combinations and to run a grid search over the combinations to evaluate the resulting models and comparing their performance. This process is known as model selection.
# MAGIC 
# MAGIC The Spark CrossValidator function performs a grid search as well as k-fold cross validation. Run the CrossValidator function with multiple values for rank and regParam:
# MAGIC 
# MAGIC class pyspark.ml.recommendation.ALS(
# MAGIC         rank=10,
# MAGIC         maxIter=10,
# MAGIC         regParam=0.1,
# MAGIC         numUserBlocks=10,
# MAGIC         numItemBlocks=10,
# MAGIC         implicitPrefs=false,
# MAGIC         alpha=1.0,
# MAGIC         userCol="user",
# MAGIC         itemCol="item",
# MAGIC         seed=None,
# MAGIC         ratingCol="rating",
# MAGIC         nonnegative=false,
# MAGIC         checkpointInterval=10,
# MAGIC         intermediateStorageLevel="MEMORY_AND_DISK",
# MAGIC         finalStorageLevel="MEMORY_AND_DISK"
# MAGIC     )

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

ratings_train, ratings_validation = ratings.randomSplit([90.0, 10.0])
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

paramGrid = ParamGridBuilder().addGrid(als.rank, [1,5,10]).addGrid(als.maxIter, [20]).addGrid(als.regParam, [0.05,0.1,0.5]).build()

crossval = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
cvModel = crossval.fit(ratings_train)   # Run cross-validation, and choose the best set of parameters.
predictions = cvModel.transform(ratings_validation)


#print "The best model is:"+ str(cvModel.getEstimatorParamMaps())
print "The root mean squared error is: " + str(evaluator.evaluate(predictions.na.drop()))

# COMMAND ----------

# MAGIC %md The more folds and parameters you add to the grid, the longer it takes to test any combination. The CrossValidator model contains more information about the performance for each combination that you can get with the avgMetrics() method. For example, you can graph the results on a plot for analysis.
# MAGIC 
# MAGIC Unfortunately, because of the SPARK-14489 issue mentioned above, the CrossValidator function can't compute the root mean squared error most of the time and provides incorrect results. You could limit this problem by making the training set much larger than the test set, but that's not a good practice. If you want to learn more about this issue, which is more a conceptual one than a technical one, and how this is being solved in the next Spark 2.2 release, you can have a look at Nick Pentreath's pull request #12896. Welcome to the Open Source world!

# COMMAND ----------

# MAGIC %md #### 4.8 Recommend movies

# COMMAND ----------

import math
def dcg (list, k):
  dcg=0
  for i in range(k):
      dcg=dcg+list[i]/math.log(i+2,2)
  return dcg

# COMMAND ----------

# MAGIC %md Use NDCG (Normalized Discounted Cumulative Gain) to evaluate the similarity of two ranking lists.

# COMMAND ----------

from pyspark.sql.functions import col

UserId=predictions.select("userId").distinct().rdd.flatMap(lambda x: x).collect()   # distinct users
NDCG=[]
for i in range(len(UserId)):
  ID=UserId[i]
  subDF=predictions.filter(col("userId")==ID).orderBy("prediction", ascending=False)
  l1 = subDF.select("rating").na.drop().rdd.flatMap(lambda x: x).collect()
  l2 = sorted(l1, reverse=True)
  k=len(l1)
  ndcg=dcg(l1,k)/dcg(l2,k)
  print ndcg
  NDCG.append(ndcg)

np.mean(NDCG)
np.max(NDCG)
np.min(NDCG)

# COMMAND ----------

predictions.createOrReplaceTempView("predictions_sql")
movies.createOrReplaceTempView("movies_sql")

# COMMAND ----------

# MAGIC %sql
# MAGIC /* See the recommend movies for user with ID=10 */
# MAGIC select p.userId, p.movieId, p.rating, p.prediction, m.title, m.genres
# MAGIC from predictions_sql p
# MAGIC join movies_sql m on p.movieId=m.movieId
# MAGIC where p.userId == 10
# MAGIC order by p.prediction desc

# COMMAND ----------

# MAGIC %md ### 5. Generalize the model to full data set.

# COMMAND ----------

# Load the full data set
movies_full = spark.read.format("csv")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load('/FileStore/tables/movies_full.csv')

ratings_full = spark.read.format("csv")\
  .option("header", "true")\
  .option("inferSchema", "true")\
  .load('/FileStore/tables/ratings_full.csv')

ratings.createOrReplaceTempView("ratings_full_sql")

# COMMAND ----------

# MAGIC %sql  /* Check the size of the ratings matrix   */
# MAGIC select *, nb_ratings/matrix_size*100 as percentage
# MAGIC from (
# MAGIC   select *, nb_users*nb_movies as matrix_size
# MAGIC   from (
# MAGIC     select count(distinct(userId)) as nb_users, count(distinct(movieId)) as nb_movies, count(*) as nb_ratings
# MAGIC     from ratings_full_sql
# MAGIC   )
# MAGIC   )
# MAGIC /* Less than 2% of the matrix is filled. */

# COMMAND ----------

ratings_full_train, ratings_full_validation = ratings.randomSplit([90.0, 10.0])
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

paramGrid = ParamGridBuilder().addGrid(als.rank, [1,5,10]).addGrid(als.maxIter, [20]).addGrid(als.regParam, [0.05,0.1,0.5]).build()

crossval = CrossValidator(estimator=als, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
cvModel_full = crossval.fit(ratings_full_train)   # Run cross-validation, and choose the best set of parameters.
predictions_full = cvModel_full.transform(ratings_full_validation)

print "The root mean squared error is: " + str(evaluator.evaluate(predictions_full.na.drop()))

# COMMAND ----------

import math
def dcg (list, k):
  dcg=0
  for i in range(k):
      dcg=dcg+list[i]/math.log(i+2,2)
  return dcg


UserId_full=predictions_full.select("userId").distinct().rdd.flatMap(lambda x: x).collect()   # distinct users
NDCG_full=[]
for i in range(len(UserId_full)):
  ID_full=UserId_full[i]
  subDF_full=predictions_full.filter(col("userId")==ID_full).orderBy("prediction", ascending=False)
  l1_full = subDF_full.select("rating").na.drop().rdd.flatMap(lambda x: x).collect()
  l2_full = sorted(l1_full, reverse=True)
  k=len(l1_full)
  ndcg_full=dcg(l1_full,k)/dcg(l2_full,k)
  print ndcg_full
  NDCG_full.append(ndcg_full)

np.mean(NDCG_full)
np.max(NDCG_full)
np.min(NDCG_full)


predictions_full.createOrReplaceTempView("predictions_full_sql")
movies_full.createOrReplaceTempView("movies_full_sql")

# COMMAND ----------

# MAGIC %sql
# MAGIC /* See the recommend movies for user with ID=10 */
# MAGIC select p.userId, p.movieId, p.rating, p.prediction, m.title, m.genres
# MAGIC from predictions_full_sql p
# MAGIC join movies_full_sql m on p.movieId=m.movieId
# MAGIC where p.userId == 10
# MAGIC order by p.prediction desc

# COMMAND ----------

# MAGIC %md Reference: https://datascience.ibm.com/exchange/public/entry/view/99b857815e69353c04d95daefb3b91fa
