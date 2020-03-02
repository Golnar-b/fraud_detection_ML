# Machine Learning Model for Fraud Detection

In this project, we investigate the fraude events related to the customers data, where data is extracted from live CRM/ Marketing platform comprising of anonymous 10K records. 
This training set is unlabelled and is sampled from a much larger dataset. 
To solve this challenge, we use Python packages from Sklearn. To complete the project, the following steps are considered.

* Data Visulaization and Engineering using PCA
* Training the model for clustering data with 3 main categories of features related to Account, Personal ID, and Redemption Activity
* Model Evaluation (performance of clustering)
* Anomaly Detection over all clsuters (i.e., detect suspicious customers that are possibly doing fraud transactions)
# Project Inputs

* Customers Database: customers_records.csv

# Project Outpus

* ML model for clustering: model/finalized_model.sav
* Dataset with labeled output as assigned to each cluster: output/clustered_data.csv
* Dataset with anomaly detection results, i.e., outliers of all clusters: output/anomaly_detected_result.csv
# Project User parameters Selection

* User can adjust the required arguments about number of clusters and update data file in parameters.yaml

# Run Instruction

1.	Install Python
2.	Clone the respository
3.	Retrieve the "data" folder
4.	Install the required libraries using the command pip install -r requirements.txt
5.	From the root folder run python -m main.py 
6.	The outputs sit inside the output folder

