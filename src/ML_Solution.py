import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import pickle


def load_clean_data(input_data_name):
    orig_data = pd.read_csv(input_data_name)
    id_count_data = orig_data.groupby('consumer_id')['consumer_id'].count().rename('count_id').reset_index()
    id_count_data['repeated_id'] = np.where(id_count_data['count_id'] > 1, 1, 0)
    orig_data = pd.merge(orig_data, id_count_data, how='inner', on=['consumer_id'])
    model_data = orig_data[['account_last_updated', 'account_age', 'app_downloads',
                        'total_offer_clicks', 'total_offer_impressions', 
                        'repeated_id', 'has_first_name', 'has_last_name', 'has_email', 
                        'avg_redemptions']]
    return(orig_data, model_data)

      