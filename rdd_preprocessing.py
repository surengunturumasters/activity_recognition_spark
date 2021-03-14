from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import asc, desc

from user_definition import *
# Do not add other libraries/packages.
sc = SparkContext.getOrCreate()
files_rdd = sc.wholeTextFiles(files)
ss = SparkSession.builder.getOrCreate()

num_files = files_rdd.count()
print(num_files)
print()

file_contents = files_rdd.map(lambda x: (x[0].split("/"), x[1])).map(lambda x: (x[0][len(x[0]) - 1], x[1]))\
        .map(lambda x: (x[0].split("_")[1:], x[1]))\
        .map(lambda x: ((x[0][0], x[0][1], x[0][2].split('.')[0]), x[1].split(";\n")))\
        .flatMapValues(lambda x: x).map(lambda x: (x[0][0], x[0][1], x[0][2], x[1].split(",")))\
        .filter(lambda x: len(x[3])==6).map(lambda x: (x[0], x[1], x[2], x[3][1], x[3][2], x[3][3], x[3][4], x[3][5]))
print(file_contents.count())
print()

schema = StructType([
    StructField('subject_id', IntegerType(), False), 
    StructField('sensor', StringType(), False), 
    StructField('device', StringType(), False), 
    StructField('activity_code', StringType(), False), 
    StructField('timestamp', LongType(), False), 
    StructField('x', FloatType(), False), 
    StructField('y', FloatType(), False), 
    StructField('z', FloatType(), False)
])
file_type_conv = file_contents.map(lambda x: (int(x[0]), x[1], x[2], x[3], int(x[4]), \
                                              float(x[5]), float(x[6]), float(x[7])))
file_df = ss.createDataFrame(file_type_conv, schema)
file_df.printSchema()
file_df = file_df.persist()

subject_files = file_df.select("subject_id").distinct().orderBy("subject_id")
subject_files.show(subject_files.count())
print()

sensor_files = file_df.select("sensor").distinct().orderBy("sensor")
sensor_files.show(sensor_files.count())
print()

activity_codes = file_df.select("activity_code").distinct().orderBy("activity_code")
activity_codes.show(activity_codes.count())

file_df.filter(f"subject_id == {int(subject_id)} and activity_code == '{str(activity_code)}'").\
        orderBy(["timestamp", "sensor"], ascending=[True, False]).show(n)
file_df.filter(f"subject_id == {int(subject_id)} and activity_code == '{str(activity_code)}'")\
        .orderBy(["timestamp", "sensor"], ascending=[True, False])\
        .withColumn("x_positive", file_df["x"] >= 0).withColumn("y_positive", file_df["y"] >= 0)\
        .withColumn("z_positive", file_df["z"] >= 0)\
        .select("subject_id", "sensor", "device", "activity_code", "timestamp", "x_positive", "y_positive"\
               , "z_positive").show(n)

ss.stop()
sc.stop()
