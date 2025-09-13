import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (adjusted_rand_score, silhouette_score, davies_bouldin_score, 
                            homogeneity_score, completeness_score, v_measure_score, 
                            calinski_harabasz_score, confusion_matrix)

def analyze_flat_clustering_results():
    """
    Provide detailed analysis of flat clustering results
    """
    # Load results
    results_df = pd.read_csv("flat_clustering_comparison_iris.csv")
    
    print("="*80)
    print("COMPREHENSIVE FLAT CLUSTERING ANALYSIS REPORT")
    print("="*80)
    
    # Display results table
    print("\n1. COMPLETE RESULTS TABLE:")
    print("-" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(results_df.round(4))
    
    # Metric rankings
    print("\n2. PERFORMANCE RANKINGS BY METRIC:")
    print("-" * 80)
    
    # For metrics where higher is better
    higher_better_metrics = ['ARI', 'Homogeneity', 'Completeness', 'V-measure', 'Silhouette', 'Calinski-Harabasz']
    for metric in higher_better_metrics:
        print(f"\n{metric} Rankings (Higher is Better):")
        valid_rows = results_df[results_df[metric].notna()]
        if not valid_rows.empty:
            sorted_rows = valid_rows.sort_values(metric, ascending=False)
            for idx, (_, row) in enumerate(sorted_rows.iterrows(), 1):
                print(f"  {idx:2d}. {row['Method']:15s}: {row[metric]:.4f}")
    
    # For Davies-Bouldin (lower is better)
    print(f"\nDavies-Bouldin Rankings (Lower is Better):")
    valid_rows = results_df[results_df['Davies-Bouldin'].notna()]
    if not valid_rows.empty:
        sorted_rows = valid_rows.sort_values('Davies-Bouldin', ascending=True)
        for idx, (_, row) in enumerate(sorted_rows.iterrows(), 1):
            print(f"  {idx:2d}. {row['Method']:15s}: {row['Davies-Bouldin']:.4f}")
    
    # Method categories analysis
    print("\n3. ANALYSIS BY CLUSTERING METHOD CATEGORY:")
    print("-" * 80)
    
    # Categorize methods
    categories = {
        'Centroid-based': ['K-Means', 'GMM'],
        'Hierarchical': ['Hierarchical'],
        'Density-based': [m for m in results_df['Method'] if 'DBSCAN' in m],
        'Graph-based': ['Spectral'],
        'Neural Network': ['GSOM', 'GSOM+DSM']
    }
    
    for category, methods in categories.items():
        print(f"\n{category}:")
        category_methods = results_df[results_df['Method'].isin(methods)]
        if not category_methods.empty:
            # Best performer in this category by ARI
            best_method = category_methods.loc[category_methods['ARI'].idxmax()]
            print(f"  Best performer: {best_method['Method']} (ARI: {best_method['ARI']:.4f})")
            
            # Average performance
            avg_ari = category_methods['ARI'].mean()
            avg_silhouette = category_methods['Silhouette'].mean()
            print(f"  Category averages: ARI={avg_ari:.4f}, Silhouette={avg_silhouette:.4f}")
    
    # Clustering stability analysis
    print("\n4. CLUSTERING STABILITY AND NOISE ANALYSIS:")
    print("-" * 80)
    
    print("Methods by number of clusters found:")
    cluster_analysis = results_df.groupby('Num_Clusters')['Method'].apply(list)
    for num_clusters, methods in cluster_analysis.items():
        print(f"  {num_clusters} clusters: {', '.join(methods)}")
    
    print("\nNoise tolerance analysis:")
    noise_methods = results_df[results_df['Noise_Points'] > 0].sort_values('Noise_Points')
    for _, row in noise_methods.iterrows():
        print(f"  {row['Method']:15s}: {row['Noise_Points']:3d} noise points ({row['Noise_Points']/150*100:.1f}%)")
    
    # Performance vs complexity trade-off
    print("\n5. PERFORMANCE VS COMPLEXITY ANALYSIS:")
    print("-" * 80)
    
    # Simple methods (K-Means, Hierarchical)
    simple_methods = ['K-Means', 'Hierarchical']
    complex_methods = ['GMM', 'Spectral'] + [m for m in results_df['Method'] if 'DBSCAN' in m]
    
    simple_avg_ari = results_df[results_df['Method'].isin(simple_methods)]['ARI'].mean()
    complex_avg_ari = results_df[results_df['Method'].isin(complex_methods)]['ARI'].mean()
    
    print(f"Simple methods (K-Means, Hierarchical) average ARI: {simple_avg_ari:.4f}")
    print(f"Complex methods (GMM, Spectral, DBSCAN) average ARI: {complex_avg_ari:.4f}")
    print(f"Complexity benefit: {complex_avg_ari - simple_avg_ari:+.4f}")
    
    # DBSCAN parameter sensitivity
    print("\n6. DBSCAN PARAMETER SENSITIVITY:")
    print("-" * 80)
    
    dbscan_methods = results_df[results_df['Method'].str.contains('DBSCAN')]
    if not dbscan_methods.empty:
        print("Parameter eps vs Performance:")
        for _, row in dbscan_methods.iterrows():
            eps_val = row['Method'].split('eps=')[1].rstrip(')')
            print(f"  eps={eps_val}: ARI={row['ARI']:.4f}, Silhouette={row['Silhouette']:.4f}, Noise={row['Noise_Points']}")
    
    # Recommendations
    print("\n7. RECOMMENDATIONS:")
    print("-" * 80)
    
    best_overall = results_df.loc[results_df['ARI'].idxmax()]
    best_balanced = results_df[(results_df['ARI'] > 0.5) & (results_df['Silhouette'] > 0.4)]
    
    print(f"Best overall performer: {best_overall['Method']} (ARI: {best_overall['ARI']:.4f})")
    
    if not best_balanced.empty:
        best_bal = best_balanced.loc[best_balanced['ARI'].idxmax()]
        print(f"Best balanced performer: {best_bal['Method']} (ARI: {best_bal['ARI']:.4f}, Silhouette: {best_bal['Silhouette']:.4f})")
    
    # GSOM analysis
    gsom_methods = results_df[results_df['Method'].isin(['GSOM', 'GSOM+DSM'])]
    if not gsom_methods.empty:
        print(f"\nGSOM Analysis:")
        print(f"GSOM methods appear to have clustering issues (creating only 1 cluster)")
        print(f"This suggests parameter tuning or algorithmic improvements may be needed")
        
        # Compare with best traditional method
        traditional_best = results_df[~results_df['Method'].isin(['GSOM', 'GSOM+DSM'])].loc[results_df[~results_df['Method'].isin(['GSOM', 'GSOM+DSM'])]['ARI'].idxmax()]
        print(f"Best traditional method ({traditional_best['Method']}) achieves ARI: {traditional_best['ARI']:.4f}")
    
    return results_df

def create_visualization():
    """
    Create comprehensive visualizations of clustering results
    """
    results_df = pd.read_csv("flat_clustering_comparison_iris.csv")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Flat Clustering Analysis - Iris Dataset', fontsize=16, fontweight='bold')
    
    # 1. ARI Comparison
    ax1 = axes[0, 0]
    methods = results_df['Method']
    ari_scores = results_df['ARI']
    bars1 = ax1.bar(range(len(methods)), ari_scores, color='skyblue', alpha=0.7)
    ax1.set_title('Adjusted Rand Index (ARI)', fontweight='bold')
    ax1.set_ylabel('ARI Score')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Highlight best performer
    best_idx = ari_scores.idxmax()
    bars1[best_idx].set_color('gold')
    
    # 2. Silhouette vs Davies-Bouldin
    ax2 = axes[0, 1]
    valid_data = results_df.dropna(subset=['Silhouette', 'Davies-Bouldin'])
    scatter = ax2.scatter(valid_data['Silhouette'], valid_data['Davies-Bouldin'], 
                         c=valid_data['ARI'], cmap='viridis', s=100, alpha=0.7)
    ax2.set_xlabel('Silhouette Score (Higher is Better)')
    ax2.set_ylabel('Davies-Bouldin Index (Lower is Better)')
    ax2.set_title('Silhouette vs Davies-Bouldin\n(Color = ARI)', fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Add text labels
    for _, row in valid_data.iterrows():
        ax2.annotate(row['Method'], (row['Silhouette'], row['Davies-Bouldin']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.colorbar(scatter, ax=ax2, label='ARI Score')
    
    # 3. Cluster Count vs Performance
    ax3 = axes[0, 2]
    scatter2 = ax3.scatter(results_df['Num_Clusters'], results_df['ARI'], 
                          c=results_df['Noise_Points'], cmap='Reds', s=100, alpha=0.7)
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('ARI Score')
    ax3.set_title('Clusters vs Performance\n(Color = Noise Points)', fontweight='bold')
    ax3.grid(alpha=0.3)
    
    for _, row in results_df.iterrows():
        ax3.annotate(row['Method'], (row['Num_Clusters'], row['ARI']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.colorbar(scatter2, ax=ax3, label='Noise Points')
    
    # 4. Multi-metric comparison (radar chart style)
    ax4 = axes[1, 0]
    metrics = ['ARI', 'Homogeneity', 'Completeness', 'V-measure']
    top_methods = results_df.nlargest(4, 'ARI')
    
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, (_, method_data) in enumerate(top_methods.iterrows()):
        values = [method_data[metric] for metric in metrics]
        ax4.bar(x + i*width, values, width, label=method_data['Method'], alpha=0.7)
    
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Score')
    ax4.set_title('Top 4 Methods: Multi-Metric Comparison', fontweight='bold')
    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. DBSCAN Parameter Analysis
    ax5 = axes[1, 1]
    dbscan_data = results_df[results_df['Method'].str.contains('DBSCAN')]
    if not dbscan_data.empty:
        eps_values = [float(method.split('eps=')[1].rstrip(')')) for method in dbscan_data['Method']]
        ax5.plot(eps_values, dbscan_data['ARI'], 'o-', label='ARI', linewidth=2, markersize=8)
        ax5.plot(eps_values, dbscan_data['Silhouette'], 's-', label='Silhouette', linewidth=2, markersize=8)
        ax5.set_xlabel('DBSCAN eps Parameter')
        ax5.set_ylabel('Score')
        ax5.set_title('DBSCAN Parameter Sensitivity', fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
    
    # 6. Method Category Performance
    ax6 = axes[1, 2]
    categories = {
        'Centroid': ['K-Means', 'GMM'],
        'Hierarchical': ['Hierarchical'],
        'Density': [m for m in results_df['Method'] if 'DBSCAN' in m],
        'Graph': ['Spectral'],
        'Neural Net': ['GSOM', 'GSOM+DSM']
    }
    
    category_scores = []
    category_names = []
    for cat_name, methods in categories.items():
        cat_data = results_df[results_df['Method'].isin(methods)]
        if not cat_data.empty:
            avg_ari = cat_data['ARI'].mean()
            category_scores.append(avg_ari)
            category_names.append(cat_name)
    
    bars6 = ax6.bar(category_names, category_scores, color='lightcoral', alpha=0.7)
    ax6.set_title('Average ARI by Method Category', fontweight='bold')
    ax6.set_ylabel('Average ARI Score')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(axis='y', alpha=0.3)
    
    # Highlight best category
    if category_scores:
        best_cat_idx = np.argmax(category_scores)
        bars6[best_cat_idx].set_color('gold')
    
    plt.tight_layout()
    plt.savefig('flat_clustering_comprehensive_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('flat_clustering_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comprehensive visualization saved as 'flat_clustering_comprehensive_analysis.pdf/png'")

def generate_detailed_report():
    """
    Generate a detailed markdown report
    """
    results_df = pd.read_csv("flat_clustering_comparison_iris.csv")
    
    report = """# Comprehensive Flat Clustering Analysis Report
## Iris Dataset Clustering Comparison

### Executive Summary
This report presents a comprehensive comparison of various flat clustering algorithms applied to the Iris dataset. The analysis includes traditional clustering methods (K-Means, Hierarchical, DBSCAN), advanced methods (GMM, Spectral), and neural network-based approaches (GSOM).

### Key Findings

"""
    
    # Best performer
    best_method = results_df.loc[results_df['ARI'].idxmax()]
    report += f"**Best Overall Performer:** {best_method['Method']} (ARI: {best_method['ARI']:.4f})\n\n"
    
    # Method rankings
    report += "### Performance Rankings (by ARI)\n\n"
    sorted_results = results_df.sort_values('ARI', ascending=False)
    for i, (_, row) in enumerate(sorted_results.iterrows(), 1):
        report += f"{i}. **{row['Method']}** - ARI: {row['ARI']:.4f}\n"
    
    report += "\n### Detailed Analysis\n\n"
    
    # Category analysis
    categories = {
        'Centroid-based Methods': ['K-Means', 'GMM'],
        'Hierarchical Methods': ['Hierarchical'],
        'Density-based Methods': [m for m in results_df['Method'] if 'DBSCAN' in m],
        'Graph-based Methods': ['Spectral'],
        'Neural Network Methods': ['GSOM', 'GSOM+DSM']
    }
    
    for category, methods in categories.items():
        report += f"#### {category}\n\n"
        cat_data = results_df[results_df['Method'].isin(methods)]
        if not cat_data.empty:
            best_in_cat = cat_data.loc[cat_data['ARI'].idxmax()]
            report += f"- **Best in category:** {best_in_cat['Method']} (ARI: {best_in_cat['ARI']:.4f})\n"
            report += f"- **Methods evaluated:** {', '.join(methods)}\n"
            report += f"- **Average ARI:** {cat_data['ARI'].mean():.4f}\n\n"
    
    # DBSCAN analysis
    dbscan_data = results_df[results_df['Method'].str.contains('DBSCAN')]
    if not dbscan_data.empty:
        report += "#### DBSCAN Parameter Sensitivity\n\n"
        report += "| eps | ARI | Silhouette | Noise Points | Clusters |\n"
        report += "|-----|-----|------------|--------------|----------|\n"
        for _, row in dbscan_data.iterrows():
            eps_val = row['Method'].split('eps=')[1].rstrip(')')
            report += f"| {eps_val} | {row['ARI']:.4f} | {row['Silhouette']:.4f} | {row['Noise_Points']} | {row['Num_Clusters']} |\n"
        report += "\n"
    
    # GSOM analysis
    gsom_data = results_df[results_df['Method'].isin(['GSOM', 'GSOM+DSM'])]
    if not gsom_data.empty:
        report += "#### GSOM Analysis\n\n"
        report += "The GSOM methods show concerning results, creating only single clusters. This suggests:\n"
        report += "- Parameter tuning may be required\n"
        report += "- The spread factor or growth threshold may need adjustment\n"
        report += "- The DSM (Distance-Spanning Method) may not be effectively separating clusters\n\n"
    
    # Recommendations
    report += "### Recommendations\n\n"
    report += f"1. **For best accuracy:** Use {best_method['Method']} (ARI: {best_method['ARI']:.4f})\n"
    
    balanced_methods = results_df[(results_df['ARI'] > 0.5) & (results_df['Silhouette'].notna()) & (results_df['Silhouette'] > 0.4)]
    if not balanced_methods.empty:
        best_balanced = balanced_methods.loc[balanced_methods['ARI'].idxmax()]
        report += f"2. **For balanced performance:** Use {best_balanced['Method']} (ARI: {best_balanced['ARI']:.4f}, Silhouette: {best_balanced['Silhouette']:.4f})\n"
    
    # Handle noise
    low_noise_methods = results_df[results_df['Noise_Points'] < 10]
    if not low_noise_methods.empty:
        best_low_noise = low_noise_methods.loc[low_noise_methods['ARI'].idxmax()]
        report += f"3. **For low noise tolerance:** Use {best_low_noise['Method']} (ARI: {best_low_noise['ARI']:.4f}, Noise: {best_low_noise['Noise_Points']} points)\n"
    
    report += "\n### Complete Results Table\n\n"
    report += results_df.to_markdown(index=False, floatfmt='.4f')
    
    # Save report
    with open('flat_clustering_analysis_report.md', 'w') as f:
        f.write(report)
    
    print("Detailed report saved as 'flat_clustering_analysis_report.md'")

if __name__ == "__main__":
    print("Starting comprehensive flat clustering analysis...")
    
    # Run analysis
    analyze_flat_clustering_results()
    
    # Create visualizations
    create_visualization()
    
    # Generate detailed report
    generate_detailed_report()
    
    print("\nAnalysis complete! Check the following files:")
    print("- flat_clustering_comparison_iris.csv (raw data)")
    print("- flat_clustering_comprehensive_analysis.pdf (visualizations)")
    print("- flat_clustering_analysis_report.md (detailed report)")
