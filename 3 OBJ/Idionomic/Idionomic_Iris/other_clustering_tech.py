"""
Comprehensive Clustering Techniques and Evaluation Metrics
Applied to the Iris Dataset for Comparison with GSOM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Clustering Algorithms
from sklearn.cluster import (
    KMeans, 
    DBSCAN, 
    AgglomerativeClustering, 
    SpectralClustering,
    MeanShift,
    AffinityPropagation,
    Birch,
    OPTICS
)
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans

# Try to import optional packages
try:
    from sklearn_extra.cluster import KMedoids
    KMEDOIDS_AVAILABLE = True
except ImportError:
    KMEDOIDS_AVAILABLE = False
    print("Warning: scikit-learn-extra not available. K-Medoids will be skipped.")

# Evaluation Metrics
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
    confusion_matrix,
    classification_report
)

# Additional utilities
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import time

class ComprehensiveClusteringAnalysis:
    """
    Comprehensive clustering analysis class that applies multiple clustering techniques
    and evaluates them using various metrics.
    """
    
    def __init__(self, data_path="Iris.csv"):
        """Initialize with dataset loading and preprocessing"""
        self.data_path = data_path
        self.results = {}
        self.load_and_prepare_data()
        
    def load_and_prepare_data(self):
        """Load and prepare the Iris dataset"""
        print("Loading and preparing Iris dataset...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Extract features and true labels
        self.feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        self.X = df[self.feature_columns].values
        self.y_true = df['Species'].astype('category').cat.codes.values
        self.species_names = df['Species'].astype('category').cat.categories.tolist()
        
        # Create different scalings for different algorithms
        self.scaler_standard = StandardScaler()
        self.scaler_minmax = MinMaxScaler()
        
        self.X_scaled = self.scaler_standard.fit_transform(self.X)
        self.X_minmax = self.scaler_minmax.fit_transform(self.X)
        
        # PCA for visualization
        self.pca = PCA(n_components=2)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        print(f"Dataset shape: {self.X.shape}")
        print(f"Number of classes: {len(np.unique(self.y_true))}")
        print(f"Class distribution: {Counter(self.y_true)}")
        print(f"Species: {self.species_names}")
        
    def apply_clustering_algorithms(self):
        """Apply various clustering algorithms"""
        print("\nApplying clustering algorithms...")
        
        algorithms = {
            # Centroid-based
            'K-Means': KMeans(n_clusters=3, random_state=42, n_init=10),
            'MiniBatch K-Means': MiniBatchKMeans(n_clusters=3, random_state=42, n_init=10),
            
            # Probabilistic
            'Gaussian Mixture': GaussianMixture(n_components=3, random_state=42),
            
            # Hierarchical
            'Agglomerative (Ward)': AgglomerativeClustering(n_clusters=3, linkage='ward'),
            'Agglomerative (Complete)': AgglomerativeClustering(n_clusters=3, linkage='complete'),
            'Agglomerative (Average)': AgglomerativeClustering(n_clusters=3, linkage='average'),
            
            # Density-based
            'DBSCAN (eps=0.5)': DBSCAN(eps=0.5, min_samples=5),
            'DBSCAN (eps=0.7)': DBSCAN(eps=0.7, min_samples=5),
            'DBSCAN (eps=1.0)': DBSCAN(eps=1.0, min_samples=5),
            'OPTICS': OPTICS(min_samples=5),
            
            # Graph-based
            'Spectral Clustering': SpectralClustering(n_clusters=3, random_state=42, n_init=10),
            
            # Other methods
            'Mean Shift': MeanShift(),
            'Affinity Propagation': AffinityPropagation(random_state=42),
            'BIRCH': Birch(n_clusters=3)
        }
        
        # Add K-Medoids if available
        if KMEDOIDS_AVAILABLE:
            algorithms['K-Medoids'] = KMedoids(n_clusters=3, random_state=42)
        
        clustering_results = {}
        timing_results = {}
        
        for name, algorithm in algorithms.items():
            print(f"  Running {name}...")
            
            try:
                start_time = time.time()
                
                # Choose appropriate data scaling
                if 'DBSCAN' in name or 'OPTICS' in name or 'Mean Shift' in name:
                    data_to_use = self.X_scaled
                elif 'Spectral' in name:
                    data_to_use = self.X_scaled
                else:
                    data_to_use = self.X_scaled
                
                # Fit the algorithm
                if 'Gaussian Mixture' in name:
                    labels = algorithm.fit_predict(data_to_use)
                else:
                    labels = algorithm.fit_predict(data_to_use)
                
                end_time = time.time()
                
                clustering_results[name] = labels
                timing_results[name] = end_time - start_time
                
            except Exception as e:
                print(f"    Error with {name}: {str(e)}")
                clustering_results[name] = np.full(len(self.X), -1)
                timing_results[name] = np.nan
        
        self.clustering_results = clustering_results
        self.timing_results = timing_results
        
        print(f"Completed {len(clustering_results)} clustering algorithms")
        
    def calculate_evaluation_metrics(self):
        """Calculate comprehensive evaluation metrics for all clustering results"""
        print("\nCalculating evaluation metrics...")
        
        metrics_data = []
        
        for method_name, labels in self.clustering_results.items():
            print(f"  Evaluating {method_name}...")
            
            # Basic cluster information
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(labels == -1)
            
            # External validation metrics (require true labels)
            try:
                ari = adjusted_rand_score(self.y_true, labels)
                nmi = normalized_mutual_info_score(self.y_true, labels)
                ami = adjusted_mutual_info_score(self.y_true, labels)
                homogeneity = homogeneity_score(self.y_true, labels)
                completeness = completeness_score(self.y_true, labels)
                v_measure = v_measure_score(self.y_true, labels)
                fm_score = fowlkes_mallows_score(self.y_true, labels)
            except:
                ari = nmi = ami = homogeneity = completeness = v_measure = fm_score = np.nan
            
            # Internal validation metrics (don't require true labels)
            try:
                if n_clusters > 1 and n_noise < len(labels):
                    # Filter out noise points for internal metrics
                    if -1 in labels:
                        non_noise_mask = labels != -1
                        if np.sum(non_noise_mask) > 1 and len(np.unique(labels[non_noise_mask])) > 1:
                            silhouette = silhouette_score(self.X_scaled[non_noise_mask], labels[non_noise_mask])
                            davies_bouldin = davies_bouldin_score(self.X_scaled[non_noise_mask], labels[non_noise_mask])
                            calinski_harabasz = calinski_harabasz_score(self.X_scaled[non_noise_mask], labels[non_noise_mask])
                        else:
                            silhouette = davies_bouldin = calinski_harabasz = np.nan
                    else:
                        silhouette = silhouette_score(self.X_scaled, labels)
                        davies_bouldin = davies_bouldin_score(self.X_scaled, labels)
                        calinski_harabasz = calinski_harabasz_score(self.X_scaled, labels)
                else:
                    silhouette = davies_bouldin = calinski_harabasz = np.nan
            except:
                silhouette = davies_bouldin = calinski_harabasz = np.nan
            
            # Timing
            execution_time = self.timing_results.get(method_name, np.nan)
            
            # Store results
            metrics_data.append({
                'Method': method_name,
                'ARI': ari,
                'NMI': nmi,
                'AMI': ami,
                'Homogeneity': homogeneity,
                'Completeness': completeness,
                'V-measure': v_measure,
                'Fowlkes-Mallows': fm_score,
                'Silhouette': silhouette,
                'Davies-Bouldin': davies_bouldin,
                'Calinski-Harabasz': calinski_harabasz,
                'N_Clusters': n_clusters,
                'N_Noise': n_noise,
                'Execution_Time': execution_time
            })
        
        self.metrics_df = pd.DataFrame(metrics_data)
        
        # Save results
        self.metrics_df.to_csv('comprehensive_clustering_metrics.csv', index=False)
        print(f"Saved metrics for {len(self.metrics_df)} methods")
        
    def analyze_results(self):
        """Perform comprehensive analysis of clustering results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE CLUSTERING ANALYSIS RESULTS")
        print("="*80)
        
        # Display complete results table
        print("\n1. COMPLETE METRICS TABLE:")
        print("-" * 80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(self.metrics_df.round(4))
        
        # Rankings by different metrics
        print("\n2. PERFORMANCE RANKINGS:")
        print("-" * 80)
        
        # ARI Rankings
        print("\nAdjusted Rand Index (ARI) Rankings:")
        ari_ranking = self.metrics_df.sort_values('ARI', ascending=False)
        for idx, (_, row) in enumerate(ari_ranking.iterrows(), 1):
            print(f"  {idx:2d}. {row['Method']:25s}: {row['ARI']:7.4f}")
        
        # Silhouette Rankings
        print("\nSilhouette Score Rankings:")
        silhouette_ranking = self.metrics_df[self.metrics_df['Silhouette'].notna()].sort_values('Silhouette', ascending=False)
        for idx, (_, row) in enumerate(silhouette_ranking.iterrows(), 1):
            print(f"  {idx:2d}. {row['Method']:25s}: {row['Silhouette']:7.4f}")
        
        # Category Analysis
        print("\n3. ALGORITHM CATEGORY ANALYSIS:")
        print("-" * 80)
        
        categories = {
            'Centroid-based': ['K-Means', 'MiniBatch K-Means'] + (['K-Medoids'] if KMEDOIDS_AVAILABLE else []),
            'Probabilistic': ['Gaussian Mixture'],
            'Hierarchical': ['Agglomerative (Ward)', 'Agglomerative (Complete)', 'Agglomerative (Average)'],
            'Density-based': ['DBSCAN (eps=0.5)', 'DBSCAN (eps=0.7)', 'DBSCAN (eps=1.0)', 'OPTICS'],
            'Graph-based': ['Spectral Clustering'],
            'Other': ['Mean Shift', 'Affinity Propagation', 'BIRCH']
        }
        
        for category, methods in categories.items():
            category_data = self.metrics_df[self.metrics_df['Method'].isin(methods)]
            if not category_data.empty:
                avg_ari = category_data['ARI'].mean()
                best_method = category_data.loc[category_data['ARI'].idxmax()]
                print(f"\n{category}:")
                print(f"  Best performer: {best_method['Method']} (ARI: {best_method['ARI']:.4f})")
                print(f"  Category average ARI: {avg_ari:.4f}")
                print(f"  Number of methods: {len(category_data)}")
        
        # Performance vs Speed Analysis
        print("\n4. PERFORMANCE vs SPEED ANALYSIS:")
        print("-" * 80)
        timing_data = self.metrics_df[self.metrics_df['Execution_Time'].notna()]
        if not timing_data.empty:
            fastest = timing_data.loc[timing_data['Execution_Time'].idxmin()]
            slowest = timing_data.loc[timing_data['Execution_Time'].idxmax()]
            
            print(f"Fastest method: {fastest['Method']} ({fastest['Execution_Time']:.4f}s, ARI: {fastest['ARI']:.4f})")
            print(f"Slowest method: {slowest['Method']} ({slowest['Execution_Time']:.4f}s, ARI: {slowest['ARI']:.4f})")
            
            # Best speed-performance trade-off
            timing_data['speed_performance_ratio'] = timing_data['ARI'] / timing_data['Execution_Time']
            best_tradeoff = timing_data.loc[timing_data['speed_performance_ratio'].idxmax()]
            print(f"Best speed-performance trade-off: {best_tradeoff['Method']} (Ratio: {best_tradeoff['speed_performance_ratio']:.2f})")
        
        # Noise Analysis
        print("\n5. NOISE TOLERANCE ANALYSIS:")
        print("-" * 80)
        noise_methods = self.metrics_df[self.metrics_df['N_Noise'] > 0].sort_values('N_Noise')
        print("Methods that identify noise points:")
        for _, row in noise_methods.iterrows():
            noise_percentage = (row['N_Noise'] / len(self.X)) * 100
            print(f"  {row['Method']:25s}: {row['N_Noise']:3d} points ({noise_percentage:5.1f}%)")
        
        # Recommendations
        print("\n6. RECOMMENDATIONS:")
        print("-" * 80)
        
        best_overall = self.metrics_df.loc[self.metrics_df['ARI'].idxmax()]
        print(f"Best overall accuracy: {best_overall['Method']} (ARI: {best_overall['ARI']:.4f})")
        
        best_internal = self.metrics_df[self.metrics_df['Silhouette'].notna()].loc[self.metrics_df[self.metrics_df['Silhouette'].notna()]['Silhouette'].idxmax()]
        print(f"Best internal validation: {best_internal['Method']} (Silhouette: {best_internal['Silhouette']:.4f})")
        
        if not timing_data.empty:
            efficient_methods = timing_data[(timing_data['ARI'] > 0.6) & (timing_data['Execution_Time'] < timing_data['Execution_Time'].median())]
            if not efficient_methods.empty:
                best_efficient = efficient_methods.loc[efficient_methods['ARI'].idxmax()]
                print(f"Best efficient method: {best_efficient['Method']} (ARI: {best_efficient['ARI']:.4f}, Time: {best_efficient['Execution_Time']:.4f}s)")
        
    def create_visualizations(self):
        """Create comprehensive visualizations of clustering results"""
        print("\nCreating comprehensive visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        
        # 1. Performance comparison chart
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Clustering Algorithm Comparison - Iris Dataset', fontsize=16, fontweight='bold')
        
        # ARI comparison
        ax1 = axes[0, 0]
        methods = self.metrics_df['Method'].str.replace(' ', '\n', regex=False)
        ari_scores = self.metrics_df['ARI']
        bars1 = ax1.bar(range(len(methods)), ari_scores, color='skyblue', alpha=0.7)
        ax1.set_title('Adjusted Rand Index (ARI)', fontweight='bold')
        ax1.set_ylabel('ARI Score')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax1.grid(axis='y', alpha=0.3)
        
        # Highlight best performer
        best_idx = ari_scores.idxmax()
        bars1[best_idx].set_color('gold')
        
        # Silhouette vs Davies-Bouldin
        ax2 = axes[0, 1]
        valid_data = self.metrics_df.dropna(subset=['Silhouette', 'Davies-Bouldin'])
        if not valid_data.empty:
            scatter = ax2.scatter(valid_data['Silhouette'], valid_data['Davies-Bouldin'], 
                                c=valid_data['ARI'], cmap='viridis', s=100, alpha=0.7)
            ax2.set_xlabel('Silhouette Score')
            ax2.set_ylabel('Davies-Bouldin Index')
            ax2.set_title('Internal Validation Metrics\n(Color = ARI)', fontweight='bold')
            ax2.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax2, label='ARI Score')
        
        # Execution time comparison
        ax3 = axes[0, 2]
        timing_data = self.metrics_df[self.metrics_df['Execution_Time'].notna()]
        if not timing_data.empty:
            timing_methods = timing_data['Method'].str.replace(' ', '\n', regex=False)
            times = timing_data['Execution_Time']
            bars3 = ax3.bar(range(len(timing_methods)), times, color='lightcoral', alpha=0.7)
            ax3.set_title('Execution Time Comparison', fontweight='bold')
            ax3.set_ylabel('Time (seconds)')
            ax3.set_xticks(range(len(timing_methods)))
            ax3.set_xticklabels(timing_methods, rotation=45, ha='right', fontsize=8)
            ax3.grid(axis='y', alpha=0.3)
        
        # Multi-metric radar-style comparison (top 5 methods)
        ax4 = axes[1, 0]
        top_methods = self.metrics_df.nlargest(5, 'ARI')
        metrics_for_radar = ['ARI', 'NMI', 'Homogeneity', 'Completeness', 'V-measure']
        
        x = np.arange(len(metrics_for_radar))
        width = 0.15
        
        for i, (_, method_data) in enumerate(top_methods.iterrows()):
            values = [method_data[metric] for metric in metrics_for_radar]
            ax4.bar(x + i*width, values, width, label=method_data['Method'][:15], alpha=0.7)
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Score')
        ax4.set_title('Top 5 Methods: Multi-Metric Comparison', fontweight='bold')
        ax4.set_xticks(x + width * 2)
        ax4.set_xticklabels(metrics_for_radar, rotation=45)
        ax4.legend(fontsize=8)
        ax4.grid(axis='y', alpha=0.3)
        
        # Cluster count vs Performance
        ax5 = axes[1, 1]
        scatter2 = ax5.scatter(self.metrics_df['N_Clusters'], self.metrics_df['ARI'], 
                              c=self.metrics_df['N_Noise'], cmap='Reds', s=100, alpha=0.7)
        ax5.set_xlabel('Number of Clusters')
        ax5.set_ylabel('ARI Score')
        ax5.set_title('Clusters vs Performance\n(Color = Noise Points)', fontweight='bold')
        ax5.grid(alpha=0.3)
        plt.colorbar(scatter2, ax=ax5, label='Noise Points')
        
        # Performance vs Speed scatter
        ax6 = axes[1, 2]
        if not timing_data.empty:
            scatter3 = ax6.scatter(timing_data['Execution_Time'], timing_data['ARI'], 
                                  s=100, alpha=0.7, c='purple')
            ax6.set_xlabel('Execution Time (seconds)')
            ax6.set_ylabel('ARI Score')
            ax6.set_title('Performance vs Speed Trade-off', fontweight='bold')
            ax6.grid(alpha=0.3)
            
            # Annotate points
            for _, row in timing_data.iterrows():
                ax6.annotate(row['Method'][:10], (row['Execution_Time'], row['ARI']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=6)
        
        plt.tight_layout()
        plt.savefig('comprehensive_clustering_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('comprehensive_clustering_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Clustering results visualization on PCA space
        self.visualize_clustering_results()
        
        print("Visualizations saved as 'comprehensive_clustering_comparison.pdf/png'")
        
    def visualize_clustering_results(self):
        """Visualize clustering results in PCA space"""
        print("Creating clustering results visualization...")
        
        # Get top 6 performing methods for visualization
        top_methods = self.metrics_df.nlargest(6, 'ARI')['Method'].tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Clustering Results Visualization (PCA Space)', fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for idx, method in enumerate(top_methods):
            ax = axes_flat[idx]
            labels = self.clustering_results[method]
            
            # Plot clustering results
            unique_labels = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    # Noise points
                    mask = labels == label
                    ax.scatter(self.X_pca[mask, 0], self.X_pca[mask, 1], 
                             c='black', marker='x', s=50, alpha=0.5, label='Noise')
                else:
                    mask = labels == label
                    ax.scatter(self.X_pca[mask, 0], self.X_pca[mask, 1], 
                             c=[color], s=50, alpha=0.7, label=f'Cluster {label}')
            
            # Get metrics for this method
            method_metrics = self.metrics_df[self.metrics_df['Method'] == method].iloc[0]
            ax.set_title(f'{method}\nARI: {method_metrics["ARI"]:.3f}, Sil: {method_metrics["Silhouette"]:.3f}')
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig('clustering_results_pca_visualization.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('clustering_results_pca_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create true labels visualization for reference
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        colors = ['red', 'green', 'blue']
        for i, species in enumerate(self.species_names):
            mask = self.y_true == i
            ax.scatter(self.X_pca[mask, 0], self.X_pca[mask, 1], 
                      c=colors[i], s=50, alpha=0.7, label=species)
        
        ax.set_title('True Species Labels (Reference)', fontweight='bold')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.savefig('true_labels_pca_reference.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('true_labels_pca_reference.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_detailed_report(self):
        """Generate a comprehensive markdown report"""
        print("Generating detailed report...")
        
        report = """# Comprehensive Clustering Techniques Analysis Report
## Iris Dataset - Multiple Algorithm Comparison

### Executive Summary
This report presents a comprehensive comparison of 15 different clustering algorithms applied to the Iris dataset. 
The analysis includes centroid-based, probabilistic, hierarchical, density-based, graph-based, and other clustering methods.

### Dataset Information
- **Dataset**: Iris flower dataset
- **Samples**: 150 instances
- **Features**: 4 (SepalLength, SepalWidth, PetalLength, PetalWidth)
- **True Classes**: 3 (Iris-setosa, Iris-versicolor, Iris-virginica)
- **Class Distribution**: Balanced (50 samples per class)

### Algorithms Evaluated

#### Centroid-based Methods
- K-Means
- MiniBatch K-Means  
- K-Medoids

#### Probabilistic Methods
- Gaussian Mixture Model (GMM)

#### Hierarchical Methods
- Agglomerative Clustering (Ward linkage)
- Agglomerative Clustering (Complete linkage)
- Agglomerative Clustering (Average linkage)

#### Density-based Methods
- DBSCAN (multiple eps values: 0.5, 0.7, 1.0)
- OPTICS

#### Graph-based Methods
- Spectral Clustering

#### Other Methods
- Mean Shift
- Affinity Propagation
- BIRCH

### Performance Results

"""
        
        # Add results table
        report += "#### Complete Performance Metrics\n\n"
        report += self.metrics_df.to_string(index=False, float_format='%.4f')
        report += "\n\n"
        
        # Add rankings
        report += "#### Performance Rankings (by ARI)\n\n"
        ari_ranking = self.metrics_df.sort_values('ARI', ascending=False)
        for idx, (_, row) in enumerate(ari_ranking.iterrows(), 1):
            report += f"{idx}. **{row['Method']}** - ARI: {row['ARI']:.4f}\n"
        
        # Add best performers by category
        categories = {
            'Centroid-based': ['K-Means', 'MiniBatch K-Means'] + (['K-Medoids'] if KMEDOIDS_AVAILABLE else []),
            'Probabilistic': ['Gaussian Mixture'],
            'Hierarchical': ['Agglomerative (Ward)', 'Agglomerative (Complete)', 'Agglomerative (Average)'],
            'Density-based': ['DBSCAN (eps=0.5)', 'DBSCAN (eps=0.7)', 'DBSCAN (eps=1.0)', 'OPTICS'],
            'Graph-based': ['Spectral Clustering'],
            'Other': ['Mean Shift', 'Affinity Propagation', 'BIRCH']
        }
        
        report += "\n#### Best Performers by Category\n\n"
        for category, methods in categories.items():
            category_data = self.metrics_df[self.metrics_df['Method'].isin(methods)]
            if not category_data.empty:
                best_method = category_data.loc[category_data['ARI'].idxmax()]
                avg_ari = category_data['ARI'].mean()
                report += f"**{category}**: {best_method['Method']} (ARI: {best_method['ARI']:.4f}, Category Avg: {avg_ari:.4f})\n\n"
        
        # Add recommendations
        best_overall = self.metrics_df.loc[self.metrics_df['ARI'].idxmax()]
        report += f"\n### Recommendations\n\n"
        report += f"**Best Overall Performance**: {best_overall['Method']} (ARI: {best_overall['ARI']:.4f})\n\n"
        
        # Add timing analysis if available
        timing_data = self.metrics_df[self.metrics_df['Execution_Time'].notna()]
        if not timing_data.empty:
            fastest = timing_data.loc[timing_data['Execution_Time'].idxmin()]
            report += f"**Fastest Method**: {fastest['Method']} ({fastest['Execution_Time']:.4f}s, ARI: {fastest['ARI']:.4f})\n\n"
        
        # Save report
        with open('comprehensive_clustering_analysis_report.md', 'w') as f:
            f.write(report)
        
        print("Detailed report saved as 'comprehensive_clustering_analysis_report.md'")
        
    def run_complete_analysis(self):
        """Run the complete clustering analysis pipeline"""
        print("Starting comprehensive clustering analysis pipeline...")
        print("="*80)
        
        # Run all analysis steps
        self.apply_clustering_algorithms()
        self.calculate_evaluation_metrics()
        self.analyze_results()
        self.create_visualizations()
        self.generate_detailed_report()
        self.compare_with_benchmarks()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated files:")
        print("- comprehensive_clustering_metrics.csv (raw metrics data)")
        print("- comprehensive_clustering_comparison.pdf/png (performance visualizations)")  
        print("- clustering_results_pca_visualization.pdf/png (clustering results in PCA space)")
        print("- true_labels_pca_reference.pdf/png (true labels reference)")
        print("- comprehensive_clustering_analysis_report.md (detailed report)")
        print("- clustering_comparison_with_benchmarks.csv (benchmark comparison)")
        
        # Return results for further analysis
        return self.metrics_df, self.clustering_results
    
    def compare_with_benchmarks(self):
        """Compare results with existing benchmarks and GSOM results"""
        print("\nComparing with benchmarks and existing results...")
        
        # Try to load existing GSOM results for comparison
        gsom_results = {}
        
        # Check if flat clustering comparison exists
        try:
            flat_results = pd.read_csv('flat_clustering_comparison_iris.csv')
            print("Found existing flat clustering results for comparison")
            
            # Extract GSOM results if available
            gsom_methods = flat_results[flat_results['Method'].isin(['GSOM', 'GSOM+DSM'])]
            if not gsom_methods.empty:
                for _, row in gsom_methods.iterrows():
                    gsom_results[row['Method']] = {
                        'ARI': row['ARI'],
                        'Silhouette': row.get('Silhouette', np.nan),
                        'Davies-Bouldin': row.get('Davies-Bouldin', np.nan)
                    }
        except FileNotFoundError:
            print("No existing flat clustering results found")
        
        # Create comprehensive comparison
        comparison_data = []
        
        # Add current results
        for _, row in self.metrics_df.iterrows():
            comparison_data.append({
                'Method': row['Method'],
                'Category': self.get_method_category(row['Method']),
                'ARI': row['ARI'],
                'Silhouette': row['Silhouette'],
                'Davies-Bouldin': row['Davies-Bouldin'],
                'Execution_Time': row['Execution_Time'],
                'N_Clusters': row['N_Clusters'],
                'N_Noise': row['N_Noise'],
                'Source': 'Current Analysis'
            })
        
        # Add GSOM results if available
        for method, metrics in gsom_results.items():
            comparison_data.append({
                'Method': method,
                'Category': 'Neural Network (GSOM)',
                'ARI': metrics['ARI'],
                'Silhouette': metrics['Silhouette'],
                'Davies-Bouldin': metrics['Davies-Bouldin'],
                'Execution_Time': np.nan,
                'N_Clusters': np.nan,
                'N_Noise': np.nan,
                'Source': 'Previous GSOM Analysis'
            })
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison results
        comparison_df.to_csv('clustering_comparison_with_benchmarks.csv', index=False)
        
        # Generate detailed comparison analysis
        self.generate_benchmark_comparison_report(comparison_df)
        
        print(f"Benchmark comparison completed with {len(comparison_df)} methods")
        
    def get_method_category(self, method_name):
        """Get category for a clustering method"""
        if any(x in method_name for x in ['K-Means', 'K-Medoids']):
            return 'Centroid-based'
        elif 'Gaussian Mixture' in method_name:
            return 'Probabilistic'
        elif 'Agglomerative' in method_name:
            return 'Hierarchical'
        elif any(x in method_name for x in ['DBSCAN', 'OPTICS']):
            return 'Density-based'
        elif 'Spectral' in method_name:
            return 'Graph-based'
        elif any(x in method_name for x in ['GSOM', 'DSM']):
            return 'Neural Network (GSOM)'
        else:
            return 'Other'
    
    def generate_benchmark_comparison_report(self, comparison_df):
        """Generate a detailed benchmark comparison report"""
        print("Generating benchmark comparison report...")
        
        report = """# Comprehensive Clustering Benchmark Comparison Report
## Iris Dataset - All Methods Comparison

### Overview
This report compares clustering results from multiple sources and provides a comprehensive benchmark analysis.

### Data Sources
"""
        
        # Add source information
        sources = comparison_df['Source'].unique()
        for source in sources:
            source_methods = comparison_df[comparison_df['Source'] == source]
            report += f"- **{source}**: {len(source_methods)} methods\n"
        
        report += f"\n### Complete Comparison Results\n\n"
        
        # Sort by ARI for better readability
        sorted_comparison = comparison_df.sort_values('ARI', ascending=False)
        
        # Create comparison table
        report += "| Rank | Method | Category | ARI | Silhouette | Davies-Bouldin | Source |\n"
        report += "|------|--------|----------|-----|------------|----------------|--------|\n"
        
        for idx, (_, row) in enumerate(sorted_comparison.iterrows(), 1):
            sil_str = f"{row['Silhouette']:.4f}" if not pd.isna(row['Silhouette']) else "N/A"
            db_str = f"{row['Davies-Bouldin']:.4f}" if not pd.isna(row['Davies-Bouldin']) else "N/A"
            report += f"| {idx} | {row['Method']} | {row['Category']} | {row['ARI']:.4f} | {sil_str} | {db_str} | {row['Source']} |\n"
        
        # Category analysis
        report += "\n### Performance by Category\n\n"
        
        category_analysis = comparison_df.groupby('Category').agg({
            'ARI': ['mean', 'max', 'count'],
            'Method': lambda x: x[comparison_df.loc[x.index, 'ARI'].idxmax()]
        }).round(4)
        
        for category in comparison_df['Category'].unique():
            cat_data = comparison_df[comparison_df['Category'] == category]
            best_method = cat_data.loc[cat_data['ARI'].idxmax()]
            avg_ari = cat_data['ARI'].mean()
            max_ari = cat_data['ARI'].max()
            count = len(cat_data)
            
            report += f"**{category}**:\n"
            report += f"- Best method: {best_method['Method']} (ARI: {best_method['ARI']:.4f})\n"
            report += f"- Average ARI: {avg_ari:.4f}\n"
            report += f"- Max ARI: {max_ari:.4f}\n"
            report += f"- Number of methods: {count}\n\n"
        
        # GSOM vs Traditional comparison
        if any('GSOM' in cat for cat in comparison_df['Category'].unique()):
            report += "### GSOM vs Traditional Methods Comparison\n\n"
            
            gsom_methods = comparison_df[comparison_df['Category'].str.contains('GSOM', na=False)]
            traditional_methods = comparison_df[~comparison_df['Category'].str.contains('GSOM', na=False)]
            
            if not gsom_methods.empty and not traditional_methods.empty:
                gsom_best = gsom_methods.loc[gsom_methods['ARI'].idxmax()]
                traditional_best = traditional_methods.loc[traditional_methods['ARI'].idxmax()]
                
                report += f"**Best GSOM method**: {gsom_best['Method']} (ARI: {gsom_best['ARI']:.4f})\n"
                report += f"**Best traditional method**: {traditional_best['Method']} (ARI: {traditional_best['ARI']:.4f})\n"
                report += f"**Performance gap**: {traditional_best['ARI'] - gsom_best['ARI']:.4f}\n\n"
                
                gsom_avg = gsom_methods['ARI'].mean()
                traditional_avg = traditional_methods['ARI'].mean()
                
                report += f"**GSOM average ARI**: {gsom_avg:.4f}\n"
                report += f"**Traditional average ARI**: {traditional_avg:.4f}\n"
                report += f"**Average gap**: {traditional_avg - gsom_avg:.4f}\n\n"
        
        # Top performers analysis
        report += "### Top 5 Overall Performers\n\n"
        top_5 = sorted_comparison.head(5)
        
        for idx, (_, row) in enumerate(top_5.iterrows(), 1):
            report += f"{idx}. **{row['Method']}** ({row['Category']})\n"
            report += f"   - ARI: {row['ARI']:.4f}\n"
            report += f"   - Source: {row['Source']}\n"
            if not pd.isna(row['Silhouette']):
                report += f"   - Silhouette: {row['Silhouette']:.4f}\n"
            if not pd.isna(row['Davies-Bouldin']):
                report += f"   - Davies-Bouldin: {row['Davies-Bouldin']:.4f}\n"
            report += "\n"
        
        # Recommendations
        report += "### Recommendations\n\n"
        
        best_overall = sorted_comparison.iloc[0]
        report += f"**Best overall method**: {best_overall['Method']} from {best_overall['Source']}\n"
        report += f"**Performance**: ARI = {best_overall['ARI']:.4f}\n\n"
        
        # Method diversity analysis
        current_methods = comparison_df[comparison_df['Source'] == 'Current Analysis']
        if len(current_methods) > 0:
            current_best = current_methods.loc[current_methods['ARI'].idxmax()]
            report += f"**Best from current analysis**: {current_best['Method']} (ARI: {current_best['ARI']:.4f})\n\n"
        
        # Save report
        with open('clustering_benchmark_comparison_report.md', 'w') as f:
            f.write(report)
        
        print("Benchmark comparison report saved as 'clustering_benchmark_comparison_report.md'")
        
        # Create comparative visualization
        self.create_comparative_visualization(comparison_df)
        
        # Print summary to console
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON SUMMARY")
        print("="*80)
        
        print(f"\nTop 3 Overall Performers:")
        for idx, (_, row) in enumerate(sorted_comparison.head(3).iterrows(), 1):
            print(f"  {idx}. {row['Method']:25s} (ARI: {row['ARI']:.4f}) - {row['Source']}")
        
        if any('GSOM' in cat for cat in comparison_df['Category'].unique()):
            gsom_methods = comparison_df[comparison_df['Category'].str.contains('GSOM', na=False)]
            if not gsom_methods.empty:
                gsom_best = gsom_methods.loc[gsom_methods['ARI'].idxmax()]
                traditional_best = comparison_df[~comparison_df['Category'].str.contains('GSOM', na=False)].loc[comparison_df[~comparison_df['Category'].str.contains('GSOM', na=False)]['ARI'].idxmax()]
                
                print(f"\nGSOM vs Traditional Comparison:")
                print(f"  Best GSOM: {gsom_best['Method']:20s} (ARI: {gsom_best['ARI']:.4f})")
                print(f"  Best Traditional: {traditional_best['Method']:20s} (ARI: {traditional_best['ARI']:.4f})")
                print(f"  Performance Gap: {traditional_best['ARI'] - gsom_best['ARI']:6.4f}")
        
        return comparison_df
    
    def create_comparative_visualization(self, comparison_df):
        """Create comprehensive comparative visualizations"""
        print("Creating comparative visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        
        # Create comprehensive comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Clustering Methods Comparison', fontsize=16, fontweight='bold')
        
        # 1. Overall ARI comparison
        ax1 = axes[0, 0]
        sorted_df = comparison_df.sort_values('ARI', ascending=True)
        
        # Color by category
        categories = sorted_df['Category'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        category_colors = dict(zip(categories, colors))
        
        bar_colors = [category_colors[cat] for cat in sorted_df['Category']]
        
        bars = ax1.barh(range(len(sorted_df)), sorted_df['ARI'], color=bar_colors, alpha=0.7)
        ax1.set_yticks(range(len(sorted_df)))
        ax1.set_yticklabels(sorted_df['Method'], fontsize=8)
        ax1.set_xlabel('Adjusted Rand Index (ARI)')
        ax1.set_title('Overall Performance Comparison (ARI)', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Highlight top performers
        top_3_indices = sorted_df.nlargest(3, 'ARI').index
        for i, bar in enumerate(bars):
            if sorted_df.iloc[i].name in top_3_indices:
                bar.set_edgecolor('gold')
                bar.set_linewidth(2)
        
        # 2. Category-wise comparison
        ax2 = axes[0, 1]
        category_stats = comparison_df.groupby('Category')['ARI'].agg(['mean', 'max', 'count']).reset_index()
        
        x_pos = np.arange(len(category_stats))
        bars2 = ax2.bar(x_pos, category_stats['mean'], 
                       color=[category_colors[cat] for cat in category_stats['Category']], 
                       alpha=0.7, label='Average ARI')
        
        # Add max values as points
        ax2.scatter(x_pos, category_stats['max'], color='red', s=50, zorder=5, label='Best in Category')
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(category_stats['Category'], rotation=45, ha='right')
        ax2.set_ylabel('ARI Score')
        ax2.set_title('Category-wise Performance', fontweight='bold')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars2, category_stats['count'])):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # 3. Source comparison (if multiple sources)
        ax3 = axes[1, 0]
        sources = comparison_df['Source'].unique()
        
        if len(sources) > 1:
            source_stats = comparison_df.groupby('Source')['ARI'].agg(['mean', 'max', 'count']).reset_index()
            
            x_pos = np.arange(len(source_stats))
            bars3 = ax3.bar(x_pos, source_stats['mean'], alpha=0.7, color='lightblue', label='Average ARI')
            ax3.scatter(x_pos, source_stats['max'], color='red', s=50, zorder=5, label='Best from Source')
            
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(source_stats['Source'], rotation=45, ha='right')
            ax3.set_ylabel('ARI Score')
            ax3.set_title('Performance by Data Source', fontweight='bold')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars3, source_stats['count'])):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'n={count}', ha='center', va='bottom', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'Single Data Source\nNo Comparison Available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Source Comparison (N/A)', fontweight='bold')
        
        # 4. Multi-metric comparison for top methods
        ax4 = axes[1, 1]
        
        # Get top 5 methods with complete metrics
        complete_metrics = comparison_df.dropna(subset=['ARI', 'Silhouette', 'Davies-Bouldin'])
        
        if not complete_metrics.empty:
            top_methods = complete_metrics.nlargest(5, 'ARI')
            
            metrics = ['ARI', 'Silhouette', 'Davies-Bouldin']
            x = np.arange(len(metrics))
            width = 0.15
            
            for i, (_, method_data) in enumerate(top_methods.iterrows()):
                # Normalize Davies-Bouldin (invert since lower is better)
                values = [
                    method_data['ARI'],
                    method_data['Silhouette'],
                    1 / (1 + method_data['Davies-Bouldin'])  # Normalized inverse
                ]
                
                ax4.bar(x + i*width, values, width, 
                       label=method_data['Method'][:15], alpha=0.7)
            
            ax4.set_xlabel('Metrics')
            ax4.set_ylabel('Score (normalized)')
            ax4.set_title('Multi-Metric Comparison (Top 5)', fontweight='bold')
            ax4.set_xticks(x + width * 2)
            ax4.set_xticklabels(['ARI', 'Silhouette', 'DB (inv)'])
            ax4.legend(fontsize=8)
            ax4.grid(axis='y', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient Data\nfor Multi-Metric Comparison', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Multi-Metric Comparison (N/A)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('comprehensive_methods_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('comprehensive_methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create GSOM-specific comparison if available
        gsom_methods = comparison_df[comparison_df['Category'].str.contains('GSOM', na=False)]
        if not gsom_methods.empty:
            self.create_gsom_comparison_visualization(comparison_df, gsom_methods)
        
        print("Comparative visualizations saved as 'comprehensive_methods_comparison.pdf/png'")
    
    def create_gsom_comparison_visualization(self, comparison_df, gsom_methods):
        """Create GSOM-specific comparison visualization"""
        print("Creating GSOM-specific comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('GSOM vs Traditional Methods Comparison', fontsize=14, fontweight='bold')
        
        # 1. GSOM vs Best Traditional methods
        ax1 = axes[0]
        
        traditional_methods = comparison_df[~comparison_df['Category'].str.contains('GSOM', na=False)]
        
        # Get best from each category
        best_traditional = traditional_methods.groupby('Category')['ARI'].idxmax()
        best_traditional_methods = traditional_methods.loc[best_traditional]
        
        # Combine with GSOM methods
        combined_data = pd.concat([gsom_methods, best_traditional_methods])
        
        # Create comparison plot
        methods = combined_data['Method']
        ari_scores = combined_data['ARI']
        colors = ['red' if 'GSOM' in cat else 'blue' for cat in combined_data['Category']]
        
        bars = ax1.bar(range(len(methods)), ari_scores, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('ARI Score')
        ax1.set_title('GSOM vs Best Traditional Methods', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, ari_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Performance gap analysis
        ax2 = axes[1]
        
        if not traditional_methods.empty:
            gsom_best = gsom_methods['ARI'].max()
            traditional_best = traditional_methods['ARI'].max()
            
            categories = ['GSOM Best', 'Traditional Best']
            values = [gsom_best, traditional_best]
            colors = ['red', 'blue']
            
            bars = ax2.bar(categories, values, color=colors, alpha=0.7)
            ax2.set_ylabel('ARI Score')
            ax2.set_title('Performance Gap Analysis', fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add gap annotation
            gap = traditional_best - gsom_best
            ax2.annotate(f'Gap: {gap:.4f}', 
                        xy=(0.5, max(values) * 0.5), 
                        ha='center', va='center', 
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('gsom_vs_traditional_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('gsom_vs_traditional_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("GSOM comparison visualization saved as 'gsom_vs_traditional_comparison.pdf/png'")

def main():
    """Main function to run comprehensive clustering analysis"""
    print("Comprehensive Clustering Techniques Analysis")
    print("Iris Dataset - Multiple Algorithm Comparison")
    print("="*60)
    
    # Initialize and run analysis
    analyzer = ComprehensiveClusteringAnalysis()
    metrics_df, clustering_results = analyzer.run_complete_analysis()
    
    return analyzer, metrics_df, clustering_results

if __name__ == "__main__":
    analyzer, metrics_df, clustering_results = main()
