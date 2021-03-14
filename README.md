# Activity Recognition
Use smartphone and smartwatch data to predict activities using Apache Spark and H20. 
Examples of activities to predict include: 
  - Sitting
  - Walking
  - Eating
  - Typing
  - Standing
  - Dribbling a Basketball, etc...

## Dataset
"WISDM Smartphone and Smartwatch Activity and Biometrics Dataset"
Raw Time Series Sensor Data

Steps taken in the Project are Below: 

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

## 2) 
  
 Data Preprocessing is done using Apache Spark
 Predictions are done using Spark ML and Sparkling Water (H20)
