from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import os
import io
import base64
from datetime import datetime

app = Flask(__name__)

# Create directory for uploads and results
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Sample data generation for demonstration
def generate_sample_data():
    # Generate synthetic biomedical data
    np.random.seed(42)
    n_samples = 100
    
    # Generate gene expression data (features)
    gene_names = [f"Gene_{i}" for i in range(50)]
    expression_data = np.random.normal(0, 1, size=(n_samples, len(gene_names)))
    
    # Generate drug response data (target)
    drug_response = 5 + 0.5 * expression_data[:, 0] - 0.7 * expression_data[:, 1] + 0.1 * expression_data[:, 2] + np.random.normal(0, 0.5, n_samples)
    
    # Create sample IDs
    sample_ids = [f"Sample_{i}" for i in range(n_samples)]
    
    # Create DataFrame
    df = pd.DataFrame(expression_data, columns=gene_names)
    df['Sample_ID'] = sample_ids
    df['Drug_Response'] = drug_response
    
    return df

# Generate and save sample data
sample_data = generate_sample_data()
sample_data.to_csv('static/uploads/sample_biomedical_data.csv', index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files and not request.form.get('use_sample'):
        return jsonify({'error': 'No file provided'})
    
    if request.form.get('use_sample') == 'true':
        df = pd.read_csv('static/uploads/sample_biomedical_data.csv')
    else:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Save uploaded file
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)
        
        # Read the data
        df = pd.read_csv(file_path)
    
    # Get analysis type
    analysis_type = request.form.get('analysis_type', 'pca')
    
    # Perform the selected analysis
    if analysis_type == 'pca':
        result = perform_pca(df)
    elif analysis_type == 'clustering':
        result = perform_clustering(df)
    elif analysis_type == 'prediction':
        result = perform_prediction(df)
    else:
        return jsonify({'error': 'Invalid analysis type'})
    
    return jsonify(result)

def perform_pca(df):
    # Extract numerical features (excluding Sample_ID and Drug_Response)
    features = df.drop(['Sample_ID', 'Drug_Response'], axis=1, errors='ignore')
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['Sample_ID'] = df['Sample_ID'].values
    if 'Drug_Response' in df.columns:
        pca_df['Drug_Response'] = df['Drug_Response'].values
    
    # Create PCA plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], 
                         c=pca_df['Drug_Response'] if 'Drug_Response' in pca_df.columns else None, 
                         cmap='viridis', 
                         alpha=0.8)
    plt.colorbar(scatter, label='Drug Response')
    plt.title('PCA of Biomedical Data')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    
    # Convert plot to base64 string
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # Calculate variance explained
    variance_explained = pca.explained_variance_ratio_
    
    # Calculate feature importance (loadings)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(2)],
        index=features.columns
    )
    top_features = loadings.abs().sum(axis=1).sort_values(ascending=False).head(10).index.tolist()
    
    return {
        'plot': plot_data,
        'variance_explained': variance_explained.tolist(),
        'top_features': top_features,
        'analysis_type': 'pca'
    }

def perform_clustering(df):
    # Extract numerical features
    features = df.drop(['Sample_ID', 'Drug_Response'], axis=1, errors='ignore')
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    # Create a DataFrame with results
    result_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    result_df['Cluster'] = clusters
    result_df['Sample_ID'] = df['Sample_ID'].values
    if 'Drug_Response' in df.columns:
        result_df['Drug_Response'] = df['Drug_Response'].values
    
    # Create clustering plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=result_df, palette='viridis', s=100, alpha=0.8)
    plt.title('K-means Clustering of Biomedical Data')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    
    # Convert plot to base64 string
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # Calculate cluster statistics
    cluster_stats = result_df.groupby('Cluster').agg({
        'Drug_Response': ['mean', 'std', 'count']
    }).reset_index() if 'Drug_Response' in result_df.columns else None
    
    if cluster_stats is not None:
        cluster_stats = cluster_stats.to_dict(orient='records')
    
    return {
        'plot': plot_data,
        'cluster_stats': cluster_stats,
        'analysis_type': 'clustering'
    }

def perform_prediction(df):
    # Check if Drug_Response column exists
    if 'Drug_Response' not in df.columns:
        return {'error': 'Drug_Response column not found in the data'}
    
    # Extract features and target
    X = df.drop(['Sample_ID', 'Drug_Response'], axis=1)
    y = df['Drug_Response']
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Feature Importance for Drug Response Prediction')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    
    # Convert plot to base64 string
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # Create scatter plot of actual vs predicted values
    plt.figure(figsize=(10, 8))
    plt.scatter(y, y_pred, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.title('Actual vs Predicted Drug Response')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot to a bytes buffer
    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format='png', dpi=100)
    buffer2.seek(0)
    
    # Convert plot to base64 string
    plot_data2 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    plt.close()
    
    # Calculate model performance metrics
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return {
        'feature_importance_plot': plot_data,
        'prediction_plot': plot_data2,
        'top_features': feature_importance.head(10)['Feature'].tolist(),
        'mse': mse,
        'r2': r2,
        'analysis_type': 'prediction'
    }

@app.route('/download_sample')
def download_sample():
    return send_file('static/uploads/sample_biomedical_data.csv', 
                     download_name='sample_biomedical_data.csv',
                     as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)