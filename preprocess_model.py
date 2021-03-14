from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder


from user_definition import *
# Please do not add anything extra than pyspark related packages/libraries.
# run with the below statement
# spark-submit --driver-class-path postgresql-42.2.18.jar --executor-memory 6g
#   --driver-memory 4g hw2.py > my_output_(1/2).txt
ss = SparkSession.builder.getOrCreate()
activity_code = ss.read.jdbc(
    url=url, table=table, properties=properties).repartition(
    'activity').cache()

# Problem 1
activity_code = ss.read.jdbc(url=url, table=table, properties=properties)\
                .repartition('activity').cache()
schema = StructType([
    StructField('subject_id', IntegerType(), False),
    StructField("sensor", StringType(), False),
    StructField("device", StringType(), False),
    StructField("activity_code", StringType(), False),
    StructField("timestamp", LongType(), False),
    StructField("x", FloatType(), False),
    StructField("y", FloatType(), False),
    StructField("z", FloatType(), False)
])
files_df = create_activity_df(ss, file_rdd(ss, files), schema)
files_df = files_df.repartitionByRange("subject_id", "sensor", "device")\
                    .cache()

# Problem 2


def check_eating_strings(x):
    for each_str in eating_strings:
        if each_str in x.lower():
            return True
    return False


filter_eat = udf(check_eating_strings, BooleanType())
activity_code.filter(filter_eat('activity')).select("code").orderBy("code")\
                .show()

# Problem 3
bool_to_int = udf(convert_to_integer, IntegerType())
joined_files = files_df.join(
    activity_code, files_df.activity_code == activity_code.code)\
        .repartition(16, "subject_id", "activity", "device", "sensor")\
        .cache()
joined_eating = joined_files.withColumn("eating", bool_to_int(
    filter_eat('activity')))\
            .orderBy('subject_id', 'timestamp', 'device', 'sensor').cache()
joined_eating.select("subject_id", "sensor", "device", "activity_code",
                     "timestamp", "x", "y", "z", "eating").show(n)

# Problem 4


def sensors(x):
    return len(x) == 2


sensor_check = udf(sensors, BooleanType())


def gyro_val(x):
    return x[1]


def accel_val(x):
    return x[0]


get_gyro = udf(gyro_val, FloatType())
get_accel = udf(accel_val, FloatType())
accel_gyro_read = joined_eating.orderBy("subject_id", "activity_code",
                                        "activity",
                                        "device", "timestamp", "sensor")\
                                .groupBy("subject_id", "activity_code",
                                         "activity",
                                         "device", "timestamp")\
                                .agg(countDistinct("sensor").alias("count"),
                                     collect_list("x").alias("xs"),
                                     collect_list("y").alias("ys"),
                                     collect_list("z").alias("zs"))\
                                .filter("count==2")\
                                .withColumn("accel_x", get_accel("xs"))\
                                .withColumn("accel_y", get_accel("ys"))\
                                .withColumn("accel_z", get_accel("zs"))\
                                .withColumn("gyro_x", get_gyro("xs"))\
                                .withColumn("gyro_y", get_gyro("ys"))\
                                .withColumn("gyro_z", get_gyro("zs"))\
                                .select("subject_id", "timestamp", "device",
                                        "activity_code",
                                        "activity", "accel_x", "accel_y",
                                        "accel_z", "gyro_x", "gyro_y",
                                        "gyro_z")
accel_gyro_read = accel_gyro_read.withColumn("eating", bool_to_int(
    filter_eat('activity')))\
                                .select("subject_id", "timestamp",
                                        "device", "eating",
                                        "activity_code", "accel_x", "accel_y",
                                        "accel_z", "gyro_x", "gyro_y",
                                        "gyro_z")\
                                .cache()
print(accel_gyro_read.count())
print()

# Problem 5
accel_gyro_read_leads = accel_gyro_read
for i in range(1, window_size+1):
    accel_gyro_read_leads = accel_gyro_read_leads\
        .withColumn(f"lead_{i}_accel_x", lead("accel_x", i)
                    .over(
                        Window.partitionBy("subject_id",
                                           "activity_code", "device")
                        .orderBy("subject_id", "activity_code", "device",
                                 "timestamp")))\
        .withColumn(f"lead_{i}_accel_y", lead("accel_y", i)
                    .over(
                        Window.partitionBy("subject_id",
                                           "activity_code", "device")
                        .orderBy("subject_id", "activity_code", "device",
                                 "timestamp")))\
        .withColumn(f"lead_{i}_accel_z", lead("accel_z", i)
                    .over(
                        Window.partitionBy("subject_id",
                                           "activity_code", "device")
                        .orderBy("subject_id", "activity_code", "device",
                                 "timestamp")))\
        .withColumn(f"lead_{i}_gyro_x", lead("gyro_x", i)
                    .over(Window.partitionBy("subject_id",
                                             "activity_code", "device")
                          .orderBy("subject_id", "activity_code",
                                   "device", "timestamp")))\
        .withColumn(f"lead_{i}_gyro_y", lead("gyro_y", i)
                    .over(Window.partitionBy("subject_id", "activity_code",
                                             "device")
                          .orderBy("subject_id", "activity_code", "device",
                                   "timestamp")))\
        .withColumn(f"lead_{i}_gyro_z", lead("gyro_z", i)
                    .over(Window.partitionBy("subject_id", "activity_code",
                                             "device")
                          .orderBy("subject_id", "activity_code",
                                   "device", "timestamp")))
accel_gyro_read_leads = accel_gyro_read_leads.cache()
accel_gyro_read_leads.orderBy("subject_id", "activity_code",
                              "device", "timestamp")\
    .drop('activity_code').show(n)

# Problem 6
si = StringIndexer(inputCol="device", outputCol="device_num")
sm = si.fit(accel_gyro_read_leads)
accel_gyro_trans = sm.transform(accel_gyro_read_leads)
ohe = OneHotEncoder(inputCol="device_num", outputCol="device"+"-onehot",
                    dropLast=False)
ohe_model = ohe.fit(accel_gyro_trans)
accel_gyro_ohe = ohe_model.transform(accel_gyro_trans).drop("device")\
    .drop("device_num").withColumnRenamed("device"+"-onehot", "device")
select_cols = ['subject_id', 'timestamp', 'device', 'accel_x', 'accel_y',
               'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
for i in range(1, window_size+1):
    select_cols += [f"lead_{i}_accel_x", f"lead_{i}_accel_y",
                    f"lead_{i}_accel_z",
                    f"lead_{i}_gyro_x",
                    f"lead_{i}_gyro_y", f"lead_{i}_gyro_z"]
accel_gyro_ohe.orderBy("subject_id", "timestamp", "device")\
            .select(select_cols).show(n)

# Problem 7
input_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
for i in range(1, window_size+1):
    input_cols += [f"lead_{i}_accel_x", f"lead_{i}_accel_y",
                   f"lead_{i}_accel_z",
                   f"lead_{i}_gyro_x", f"lead_{i}_gyro_y", f"lead_{i}_gyro_z"]
va = VectorAssembler(outputCol='features', inputCols=input_cols,
                     handleInvalid="skip")
feature_data = va.transform(accel_gyro_ohe)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=True)
scaleModel = scaler.fit(feature_data)
feature_scale_data = scaleModel.transform(feature_data)\
            .orderBy("subject_id", "activity_code", "device", "timestamp")\
            .select("eating", "device", "scaledFeatures")
feature_scale_data.withColumnRenamed("scaledFeatures", "features").show(n)

# Problem 8
va2 = VectorAssembler(outputCol="features",
                      inputCols=["scaledFeatures", "device"],
                      handleInvalid="skip")
added_features = va2.transform(feature_scale_data)\
            .withColumnRenamed("eating", "label")\
            .select("features", "label")

# Problem 9
splits = added_features.randomSplit([0.8, 0.2], 1)
data_train = splits[0].cache()
data_valid = splits[1].cache()
data_train.show(n)
data_valid.show(n)

# Problem 10
lr = LogisticRegression()
bceval = BinaryClassificationEvaluator()
bceval.setMetricName("areaUnderROC")
cv = CrossValidator().setEstimator(lr).setEvaluator(bceval).setNumFolds(n_fold)
paramGrid = ParamGridBuilder().addGrid(lr.maxIter, max_iter)\
                              .addGrid(lr.regParam, reg_params).build()
cv.setEstimatorParamMaps(paramGrid)
cvmodel = cv.fit(data_train)

print(cvmodel.bestModel.coefficients)
print()
print(cvmodel.bestModel.intercept)
print()
print(cvmodel.bestModel.getMaxIter())
print()
print(cvmodel.bestModel.getRegParam())
print()

# Problem 11
print(BinaryClassificationEvaluator().setMetricName("areaUnderROC")
      .evaluate(cvmodel.bestModel.transform(data_valid)))

ss.stop()
