___
# 🌐 Extending GSOM for Unsupervised Learning-Based Modeling of Hidden Patterns in Data  
**Hierarchical Clustering. Data Skeleton Modelling. Idionomic Featuring.**

---

## 📚 Table of Contents

1. [Abstract](#abstract)  
2. [Related Works](#related-works)  
3. [Methodology](#methodology)  
4. [Experiment Setup & Implementation](#experiment-setup--implementation)  
5. [Results & Analysis](#results--analysis)  
6. [Conclusion](#conclusion)  
7. [Publications](#publications)  
8. [Links](#links)

---

## ✨ Abstract

This project enhances the **Growing Self-Organizing Map (GSOM)** by integrating **Data Skeleton Modeling (DSM)** to build a more adaptive and interpretable tool for unsupervised learning. GSOM's dynamic architecture overcomes the rigidity of traditional SOMs, and DSM adds meaningful structure to clustered data, allowing for better visualization and feature extraction.

The project evaluates GSOM+DSM on benchmark and real-world datasets, with plans to compare it against popular clustering algorithms such as K-Means, DBSCAN, GMM, and SOM. Applications in biomedical data, behavioral analytics, and text mining are being explored to demonstrate its practical impact.

> 🚧 _This project is ongoing. Results will be added upon completion of the experimental phase._

---

## 📖 Related Works

- **Self-Organizing Map (SOM):** A powerful yet limited tool for visualizing and clustering high-dimensional data due to its fixed grid structure.
- **Growing SOM (GSOM):** Enhances SOM with growth capabilities based on data complexity, improving topology preservation and flexibility.
- **Data Skeleton Modeling (DSM):** Strengthens cluster interpretation by revealing structural skeletons, particularly valuable in high-dimensional datasets.

---

## 🔬 Methodology

We approach this project in three key phases:

### 1. Theoretical Validation
- Compare GSOM vs SOM using:
  - Topographic Error (TE)  
  - Topographic Product (TP)  
  - Zrehen Measure (ZM)  
  - C-measure  

### 2. Experimental Evaluation
- Cluster performance evaluation using:
  - Silhouette Score  
  - Davies-Bouldin Index  
  - Cluster Purity  

### 3. DSM Integration
- Build a DSM pipeline to extract skeletal features from GSOM clusters
- Focus on idionomic features and hierarchical patterns

---

## ⚙️ Experiment Setup & Implementation

**Datasets:**
- UCI Repository datasets (Zoo, Iris, etc.)
- Real-world datasets from Kaggle and open data sources

**Tools & Libraries:**
- Python (NumPy, Pandas, SciPy)
- [Pygsom](https://github.com) for GSOM core
- `Bigtree` for hierarchical visualizations
- `Matplotlib`, `Plotly` for analysis plots

**Current Development Includes:**
- GSOM + DSM integration  
- Skeleton extraction and visualization module  
- Topology evaluation module  
- Benchmarking framework setup

---

## 📊 Results & Analysis

🚧 _Experiments and evaluations are currently in progress._  
Once completed, this section will present:

- GSOM’s topology preservation vs SOM  
- Hierarchical clustering performance  
- Comparison with K-Means, DBSCAN, GMM  
- Visualization of extracted data skeletons  
- Application to real-world biomedical and behavioral data

---

## 🧠 Conclusion

This research aims to establish GSOM as a more robust and interpretable alternative to traditional clustering methods by integrating DSM. Once validated, GSOM+DSM could be a valuable tool for discovering hidden patterns in complex, high-dimensional data.

Future directions include:
- Combining GSOM with neural networks for deep clustering  
- Scaling to large datasets  
- Real-world deployment in domains such as healthcare, social science, and NLP

---

## 📄 Publications

_Publications related to this project will be added here once available._

---
## 👥 Team

- **E/19/124**: Hirushi Gunasekara – [e19124@eng.pdn.ac.lk](mailto:e19124@eng.pdn.ac.lk)  
- **E/19/324**: Bimbara Rathnayake – [e19324@eng.pdn.ac.lk](mailto:e19324@eng.pdn.ac.lk)  

### 🎓 Supervisors

- **Dr. Damayanthi Herath** – [damayanthiherath@eng.pdn.ac.lk](mailto:damayanthiherath@eng.pdn.ac.lk)  
- **Prof. Damminda Alahakoon** – [D.Alahakoon@latrobe.edu.au](mailto:D.Alahakoon@latrobe.edu.au)

---

## 🔗 Links

- 🔬 [Project Repository](https://github.com/cepdnaclk/e19-4yp-Extending-the-Growing-Self-Organizing-Map-for-Unsupervised-Learning-Based-Modelling)  
- 🌐 [Project Page](https://cepdnaclk.github.io/e19-4yp-Extending-the-Growing-Self-Organizing-Map-for-Unsupervised-Learning-Based-Modelling/)  
- 🏛️ [Department of Computer Engineering, University of Peradeniya](https://eng.pdn.ac.lk/cpe/)

---

> 🛠️ _This project is a final-year research endeavor by undergraduate students at the Department of Computer Engineering, University of Peradeniya._
