# Machine Learning Model for Fraud Detection

In this project, data is extracted from live CRM/ Marketing platform comprising of anonymous 10K records. This feature/Training set is sampled from a much larger training set for Machine Learning model training. To solve this challenge you are free to use any Python packages from available ML toolkits such as Sklearn, H2O, Tensorflow etc. or Java, Scala.
This is unlabeled or unclassified data using which we provide an approach to:
 
* Train model for clustering data with 3 main categories of features related to Account, Personal ID, and Redemption Activity 
* Model Evaluation (clustering performance)
* Anomaly Detection over all clsuters (i.e., detect suspicious customers that are possibly doing fraud transactions)
 
# Project Inputs

* Customers Database: customers_records.csv

# Project Outpus

* ML model for clustering: model/finalized_model.sav
* data with labeled output as assigned to each cluster: output/clustered_data.csv
* data with anomaly detection results, i.e., outliers of all clusters: output/anomaly_detected_result.csv

# Project User parameters Selection

* User can adjust the required arguments about number of clusters for clustering model

# Run Instruction

1.	Install Python
2.	Clone the respository
3.	Retrieve the "data" folder
4.	Install the required libraries using the command pip install -r requirements.txt
5.	From the root folder run python -m main.py 
6.	The outputs sit inside the output folder

