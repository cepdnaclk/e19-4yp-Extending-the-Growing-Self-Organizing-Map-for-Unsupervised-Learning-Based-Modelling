# Crohn's Disease GSOM Analysis - Implementation Summary

## Overview
Successfully modified the GSOM analysis code to run on the Crohn's disease genetic dataset with 100 samples from each class (Healthy vs. Crohn's).

## Dataset Configuration
- **Original Dataset**: 387 samples with 213 features
  - Healthy (Class 0): 243 samples
  - Crohn's (Class 2): 144 samples
- **Processed Dataset**: 200 samples with 206 genetic features
  - Healthy: 100 samples (balanced)
  - Crohn's: 100 samples (balanced)
- **Features**: 206 genetic loci (loc1.a1, loc1.a2, ..., loc103.a1, loc103.a2)

## Key Modifications Made

### 1. Dataset Configuration Addition
Added Crohn's dataset configuration to `dataset_configs`:
```python
'crohn': {
    'file': 'crohn.csv',
    'index_col': 'id',
    'label_col': 'crohn',
    'weight_columns': None,  # Auto-detected genetic features
    'dimensions': 206,  # Detected automatically
    'max_clusters': 2,  # Healthy vs Crohn's
    'distance': 'euclidean',
    'distance_threshold': 0.4,
    'max_radius': 5
}
```

### 2. Balanced Sampling Implementation
- Automatically sampled 100 samples from each class for balanced analysis
- Used reproducible random seed (42) for consistent results
- Converted class labels to descriptive strings ('Healthy', 'Crohns')

### 3. Genetic Feature Auto-Detection
- Automatically detected all genetic feature columns (loc*.a1, loc*.a2)
- Filtered out non-genetic columns (pid, fid, mid, sex)
- Dynamic dimension calculation based on detected features

### 4. Index Column Flexibility
- Updated all methods to accept configurable index column parameter
- Fixed hardcoded 'Id' references to use dataset-specific index column ('id' for Crohn's)
- Enhanced predict method to handle weight columns correctly

### 5. Method Parameter Updates
Updated method signatures to support flexible index columns:
- `detect_outliers(..., index_col='Id')`
- `analyze_region(..., index_col='Id')`
- `compute_quantize_difference(..., index_col='Id')`
- `identify_boundary_points(..., index_col='Id')`

## Results Generated

### Analysis Output Files
1. **output_crohn.csv** - GSOM node assignments and predictions
2. **boundary_points_analysis_crohn.txt** - Detailed boundary point analysis
3. **gsom_regional_analysis_crohn.txt** - Comprehensive regional analysis

### Visualization Files (PNG format)
1. **gsom_analysis_original_data_scatter_crohn.png** - Original data distribution
2. **gsom_analysis_regional_map_crohn.png** - GSOM spatial structure with regions
3. **gsom_analysis_confusion_matrix_crohn.png** - Regional confusion matrix
4. **gsom_analysis_regional_purity_crohn.png** - Region purity analysis
5. **gsom_analysis_quantize_difference_crohn.png** - Node quantization quality
6. **gsom_analysis_outlier_analysis_crohn.png** - Outlier detection summary
7. **gsom_analysis_boundary_features_crohn.png** - Boundary point feature analysis

## Key Findings

### Regional Analysis
- **Node Count**: 996 nodes generated
- **Active Nodes**: 138 nodes with data assignments
- **Regions Identified**: 2 main regions (both mixed)
  - Region 1: 52.2% purity (44 Healthy, 48 Crohn's)
  - Region 3: 51.9% purity (56 Healthy, 52 Crohn's)

### Classification Performance
- **Overall Accuracy**: 22.0%
- **Pure Nodes**: 125 nodes (single class assignments)
- **Mixed Nodes**: 13 nodes (multiple class assignments)

### Boundary Analysis
- **Boundary Points**: 120 samples identified as class boundary cases
- **Top Discriminative Features**: Genetic loci showing highest variability
- **Boundary Score Range**: 0.352 - 0.381 (indicating moderate class overlap)

## Technical Implementation
- **Distance Metric**: Euclidean distance for genetic similarity
- **Training Parameters**: 15 growing iterations, 5 smoothing iterations
- **Normalization**: MinMax scaling for genetic feature values
- **Cluster Detection**: Automatic segmentation with distance-based separation

## Usage
To run the analysis:
```bash
python "code - Copy.py"
```

The code automatically:
1. Loads and preprocesses the Crohn's dataset
2. Samples 100 cases from each class
3. Trains the GSOM model
4. Generates comprehensive analysis and visualizations
5. Saves all results with appropriate file naming

## Code Robustness
- Flexible dataset configuration system
- Automatic feature detection
- Configurable index column handling
- Balanced sampling with reproducible seeds
- Comprehensive error handling for dimension mismatches
