from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from pyspark.sql import SparkSession
from pysparkling import H2OContext
from builtins import round

# spark-submit --driver-class-path postgresql-42.2.18.jar --executor-memory 6g
#   --driver-memory 4g hw5.py > my_output_(1/2).txt

from user_definition import *

ss = SparkSession.builder\
        .config('spark.ext.h2o.log.level', 'FATAL')\
        .getOrCreate()
sc = ss.sparkContext.getOrCreate()

hc = H2OContext.getOrCreate()

# Question 1
train_df = ss.read.parquet(train_folder)
valid_df = ss.read.parquet(valid_folder)
train_h2o = hc.asH2OFrame(train_df, "train")
valid_h2o = hc.asH2OFrame(valid_df, "valid")
train_h2o["label"] = train_h2o["label"].asfactor()  # make last label enum
valid_h2o["label"] = valid_h2o["label"].asfactor()
for i in train_h2o.columns:
    print(f"{i} - {train_h2o.types[i]}")

# Question 2
feature_preds = valid_h2o.columns[:-1]
response = valid_h2o.columns[-1]
for i in feature_preds:
    print(i)

# Question 3
# Train an XGBoost Model
xgboost_mod = H2OXGBoostEstimator(nfolds=n_fold,
                                  max_runtime_secs=max_runtime_secs,
                                  seed=seed)

xgboost_mod.train(x=feature_preds,
                  y=response,
                  training_frame=train_h2o,
                  validation_frame=valid_h2o)

print(round(xgboost_mod.auc(valid=True), n_digits))
print(xgboost_mod.confusion_matrix(valid=True))

# Question 4
# Train a Deep Learning Model
deep_mod = H2ODeepLearningEstimator(nfolds=n_fold,
                                    variable_importances=True,
                                    loss="Automatic",
                                    max_runtime_secs=max_runtime_secs,
                                    seed=seed)
deep_mod.train(x=feature_preds,
               y=response,
               training_frame=train_h2o,
               validation_frame=valid_h2o)

print(round(deep_mod.auc(valid=True), n_digits))
print(deep_mod.confusion_matrix(valid=True))

ss.stop()

