# Activity Recognition
Use smartphone and smartwatch data to predict activities using Apache Spark and H20. 
Examples of activities to predict include: 
  - Sitting
  - Walking
  - Eating
  - Typing
  - Standing
  - Dribbling a Basketball, etc...

Data Preprocess Done in Apache Spark
Analysis Using SparkML and H20

## Dataset
"WISDM Smartphone and Smartwatch Activity and Biometrics Dataset"

Raw Time Series Sensor Data

Description of dataset described [here](https://github.com/surengunturumasters/activity_recognition_spark/blob/main/WISDM-dataset-description.pdf)


**Steps taken in the Project are Below: **

## 1) Feature Extraction

Extract features used to predict the above activities:
  - subject_id
  - SmartPhone or SmartWatch Data?
  - device type: accelerometer and/or gyroscope
  - timestamp 
  - x, y, and z coordinates
Extract labels: 
  - activity code representing each activity

Organize all this information into a spark RDD and then to a Spark DataFrame

Code [here](https://github.com/surengunturumasters/activity_recognition_spark/blob/main/feature_extraction.py)

## 2) Data Preprocess
  
For each activity, included all percentile readings of: 
  - x, y, z coordinates

Order by timestamp

Code [here]()
