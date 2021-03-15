from pyspark.sql import *
from pyspark.ml import *

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline

from builtins import round

from user_definition import *

# spark-submit --driver-class-path postgresql-42.2.18.jar --executor-memory 6g
#   --driver-memory 4g hw4.py > my_output_(1/2).txt

ss = SparkSession.builder.getOrCreate()
sc = ss.sparkContext.getOrCreate()

# Problem 1
train_df = ss.read.parquet(train_folder)
test_df = ss.read.parquet(valid_folder)
print(train_df.count())
print()
print(test_df.count())
print()

# Problem 2
print("RandomForestClassifier")
rf = RandomForestClassifier()
evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderROC")

paramGrid = ParamGridBuilder().addGrid(rf.numTrees, num_trees)\
    .build()
cv = CrossValidator(estimator=rf, evaluator=evaluator,
                    numFolds=n_fold, estimatorParamMaps=paramGrid)
cvmodel = cv.fit(train_df)
print(cvmodel.bestModel.getNumTrees)
test_predicts = cvmodel.bestModel.transform(test_df)
print(str(round(evaluator.evaluate(test_predicts), n_digits)))
print()

# Problem 3
print("GBTClassifier")
gb = GBTClassifier()
evaluator = BinaryClassificationEvaluator().setMetricName("areaUnderROC")

paramGrid = ParamGridBuilder().addGrid(gb.maxDepth, max_depth)\
                            .build()
cv = CrossValidator(estimator=gb, evaluator=evaluator, numFolds=n_fold,
                    estimatorParamMaps=paramGrid)
cvmodel = cv.fit(train_df)
print(cvmodel.bestModel.getMaxDepth())
test_predicts = cvmodel.bestModel.transform(test_df)
print(str(round(evaluator.evaluate(test_predicts), n_digits)))

ss.stop()
