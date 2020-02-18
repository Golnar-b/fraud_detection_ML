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


def do_clusters_elbow_plot(model_data):
    #running a k-means model on clusters varying from 1 to 10 and generate an elbow curve
    X_scaled = MinMaxScaler().fit_transform(np.array(model_data).astype(np.float))
    
    clust = range(1, 10)
    kmeans_all = [KMeans(n_clusters=i, random_state=42) for i in clust]
    elbow_score = [kmeans_all[i].fit(X_scaled).score(X_scaled) for i in range(len(kmeans_all))]
    
    plt.plot(clust, elbow_score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()


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
    
    
def load_model_do_evaluate_cluster(model_data, n_clusters, X_scaled_reduced, PCA_df, model_filename):
    #load model
    kmeans = pickle.load(open(model_filename, 'rb'))
    
    #compute cluster centers and predict cluster indices
    X_clusters_centers = kmeans.cluster_centers_
    clust_labels = kmeans.predict(X_scaled_reduced)
    
    #evaluate clustering output
    score = silhouette_score (X_scaled_reduced, clust_labels, metric='euclidean')
    print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
        
    #add labels to model_data
    model_clustered_data = model_data
    model_clustered_data.insert((model_clustered_data.shape[1]),'kmeans', pd.DataFrame(clust_labels))

    #plot the clusters with scatter digram
    LABEL_COLOR_MAP = {0 : 'g', 1 : 'r', 2:'b', 3: 'c', 4: 'k', 5: 'y', 6: 'w', 7:'m'}
    label_color = [LABEL_COLOR_MAP[l] for l in clust_labels]
    plt.figure(figsize = (9, 7))
    plt.scatter(X_scaled_reduced[:, 0], X_scaled_reduced[:, 1], c= label_color, alpha=0.5)
    plt.xlabel(PCA_df[1][0]) 
    plt.ylabel(PCA_df[1][1])
    plt.show()
    return(model_clustered_data, X_clusters_centers, clust_labels)
    
    
def detect_anomaly_save_result(X_scaled_reduced, X_clusters_centers, clust_labels, orig_data, output_file_name):
    #calculate the distance to the cluster centroid for each point
    distance = [np.linalg.norm(x-y) for x,y in zip(X_scaled_reduced, X_clusters_centers[clust_labels])]
    
    #create predictions based on distance
    km_y_pred = np.array(distance)
    km_y_pred[distance>=np.percentile(distance, 95)] = 1
    km_y_pred[distance<np.percentile(distance, 95)] = 0
    km_y_pred.sum()
    
    data_init = orig_data.reset_index()
    data_pred = pd.DataFrame(km_y_pred).reset_index()
    data_pred.columns=['index', 'anomaly_detected']
    result = pd.merge(data_init, data_pred, on='index')
    
    result.to_csv(output_file_name)
    
    
def run_model(n_clusters):
    orig_data, model_data = load_clean_data('../data/customers_records.csv')
    do_clusters_elbow_plot(model_data)
    X_scaled_reduced, PCA_df = scale_data_select_features(model_data)
    train_and_save_model(X_scaled_reduced, n_clusters, '../model/finalized_model.sav')
    model_clustered_data, X_clusters_centers, clust_labels = load_model_do_evaluate_cluster(model_data, n_clusters, X_scaled_reduced, PCA_df, '../model/finalized_model.sav')   
    model_clustered_data.to_csv('../output/clustered_data.csv')
    detect_anomaly_save_result(X_scaled_reduced, X_clusters_centers, clust_labels, orig_data, '../output/anomaly_detected_result.csv')
