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


def scale_data_select_features(model_data):
    X = np.array(model_data).astype(np.float)
    X_scaled = StandardScaler().fit_transform(X)
    
    #get the most important features with pca
    pca = PCA(n_components=7)
    pca_model = pca.fit(X_scaled)
    x_7d = pca.transform(X_scaled)
    
    #get the index of the most important feature on each component 
    n_pcs= pca_model.components_.shape[0]
    most_important = [np.abs(pca_model.components_[i]).argmax() for i in range(n_pcs)]
    initial_feature_names = model_data.columns.tolist()
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
    dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}
    PCA_df = pd.DataFrame(sorted(dic.items()))
    
    X_scaled_reduced = x_7d
    return(X_scaled_reduced, PCA_df)
    
    
def train_and_save_model(X_scaled_reduced, n_clusters, model_filename):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled_reduced)
    pickle.dump(kmeans, open(model_filename, 'wb'))