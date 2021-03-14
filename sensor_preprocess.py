from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import *

from user_definition import *

# Do not add any other libraies/packages

# run with the below statement
# spark-submit --driver-class-path postgresql-42.2.18.jar --executor-memory 6g
#   --driver-memory 4g hw2.py > my_output_(1/2).txt
ss = SparkSession.builder.getOrCreate()
activity_code = ss.read.jdbc(
    url=url, table=table, properties=properties).repartition(
    'activity').cache()

# Problem 1
print(activity_code.select("activity").distinct().count())
print()


# Problem 2
activity_code.orderBy("activity", ascending=False).show(truncate=False)


# Problem 3
def eating(x):
    for each_str in eating_strings:
        if each_str in x.lower():
            return True
    return False


eating_udf = udf(eating, BooleanType())
eating_activity = activity_code.withColumn("eating", eating_udf("activity"))
eating_activity.printSchema()
eating_activity.orderBy("eating", "code", ascending=[False, True]).show()


# Problem 4
schema = StructType([
    StructField("subject_id", IntegerType(), False),
    StructField("sensor", StringType(), False),
    StructField("device", StringType(), False),
    StructField("activity_code", StringType(), False),
    StructField("timestamp", LongType(), False),
    StructField("x", FloatType(), False),
    StructField('y', FloatType(), False),
    StructField('z', FloatType(), False)
])
files_df = create_activity_df(ss, file_rdd(ss, files), schema)
files_df.repartitionByRange('subject_id', 'sensor', 'device').cache()
activity_files = files_df.groupBy("subject_id", "sensor", "device").agg(
    countDistinct('activity_code').alias("count")).orderBy(
    "subject_id", "device", "sensor", ascending=[1, 1, 1])
activity_files.show(activity_files.count())


# Problem 5
joined_files = files_df.join(
    activity_code, files_df.activity_code == activity_code.code)
joined_files.repartition(16, "subject_id", "activity", "device", "sensor")
joined_files.cache().first()
joined_files.groupBy("subject_id", "activity", "device", "sensor").agg(
    min('x').alias('x_min'), min('y').alias('y_min'), min('z').alias('z_min'),
    avg('x').alias('x_avg'), avg('y').alias('y_avg'), avg('z').alias('z_avg'),
    max('x').alias('x_max'), max('y').alias('y_max'),
    max('z').alias('z_max'), expr('percentile(x, array(0.05))')[0].alias(
        'x_05%'),
    expr('percentile(y, array(0.05))')[0].alias('y_05%'),
    expr('percentile(z, array(0.05))')[0].alias('z_05%'),
    expr('percentile(x, array(0.25))')[0].alias('x_25%'),
    expr('percentile(y, array(0.25))')[0].alias('y_25%'),
    expr('percentile(z, array(0.25))')[0].alias('z_25%'),
    expr('percentile(x, array(0.5))')[0].alias('x_50%'),
    expr('percentile(y, array(0.5))')[0].alias('y_50%'),
    expr('percentile(z, array(0.5))')[0].alias('z_50%'),
    expr('percentile(x, array(0.75))')[0].alias('x_75%'),
    expr('percentile(y, array(0.75))')[0].alias('y_75%'),
    expr('percentile(z, array(0.75))')[0].alias('z_75%'),
    expr('percentile(x, array(0.95))')[0].alias('x_95%'),
    expr('percentile(y, array(0.95))')[0].alias('y_95%'),
    expr('percentile(z, array(0.95))')[0].alias('z_95%'),
    stddev('x').alias('x_std'),
    stddev('y').alias('y_std'), stddev('z').alias('z_std')).orderBy(
    "activity", "subject_id", "device", "sensor").show(n)


# Problem 6
def include_activity(x):
    return activity_string in x.lower()


activity_check = udf(include_activity, BooleanType())

joined_files.filter(f"subject_id=='{subject_id}'").filter(activity_check(
    'activity')).select(
    'activity', "timestamp", "device", "sensor", "x", "y", "z")\
    .orderBy("timestamp", "device", "sensor").show(n)


# Problem 7
def sensors(x):
    return len(x) == 2


sensor_check = udf(sensors, BooleanType())


def gyro_val(x):
    return x[0]


def accel_val(x):
    return x[1]


get_gyro = udf(gyro_val, FloatType())
get_accel = udf(accel_val, FloatType())

joined_files.filter(f"subject_id=='{subject_id}'").filter(
    activity_check('activity')).withColumn(
    'sensors', collect_list('sensor').over(
        Window.partitionBy('activity', 'timestamp', 'device').orderBy(
            'activity'))).filter(
            sensor_check('sensors')).withColumn(
    'xs', collect_list('x').over(
        Window.partitionBy('activity', 'timestamp', 'device').orderBy(
            'activity'))).withColumn(
    'ys', collect_list('y').over(
        Window.partitionBy('activity', 'timestamp', 'device').orderBy(
            'activity'))).withColumn(
    'zs', collect_list('z').over(
        Window.partitionBy('activity', 'timestamp', 'device').orderBy(
            'activity'))).withColumn(
    "accel_x", get_accel("xs")).withColumn("accel_y", get_accel(
        "ys")).withColumn(
    "accel_z", get_accel("zs")).withColumn("gyro_x", get_gyro(
        "xs")).withColumn(
    "gyro_y", get_gyro("ys")).withColumn("gyro_z", get_gyro(
        "zs")).select(
    'activity_code', 'device', 'timestamp', 'accel_x', 'accel_y',
    'accel_z', "gyro_x", "gyro_y", "gyro_z").distinct().orderBy(
    'timestamp').show(n)


ss.stop()
