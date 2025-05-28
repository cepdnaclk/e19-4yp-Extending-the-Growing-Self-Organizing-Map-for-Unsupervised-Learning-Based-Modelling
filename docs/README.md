---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: eYY-4yp-project-template
title:
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

# Extending the Growing Self Organizing Map (GSOM) for Unsupervised Learning-Based Modelling of Hidden Structures and Patterns in Data

#### Team

- E/19/124, Hirushi Gunasekara, [e19124@eng.pdn.ac.lk]()
- E/19/324, Bimbara Rathnayake, [e19324@eng.pdn.ac.lk]()

#### Supervisors

- Dr. Damayanthi Herath, [damayanthiherath@eng.pdn.ac.lk]()
- Prof. Damminda Alahakoon, [D.Alahakoon@latrobe.edu.au]()

#### Table of content

1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)

---

<!-- 
DELETE THIS SAMPLE before publishing to GitHub Pages !!!
This is a sample image, to show how to add images to your page. To learn more options, please refer [this](https://projects.ce.pdn.ac.lk/docs/faq/how-to-add-an-image/)
![Sample Image](./images/sample.png) 
-->


## Abstract
This research extends the Growing Self-Organizing Map (GSOM) by integrating Data Skeleton Modeling (DSM) to enhance topology preservation, hierarchical clustering, and individual feature modeling for unsupervised learning. GSOM addresses limitations of traditional Self-Organizing Maps (SOM), such as fixed grid structures and topological distortions, by dynamically adapting to data complexity. The study validates GSOM's effectiveness using topology preservation metrics (topographic error, Zrehen measure) and benchmarks its performance against K-Means, HDBSCAN, and deep learning-based models. Real-world applications in biomedical data analysis, text mining, and behavioral analytics are explored to demonstrate GSOM's scalability and interpretability.

## Related works
The Self-Organizing Map (SOM), introduced by Kohonen, is a widely used tool for clustering and visualization but is constrained by its fixed grid structure and susceptibility to topological distortions. The Growing Self-Organizing Map (GSOM) improves upon SOM by dynamically adapting to data complexity, enhancing topology preservation and enabling hierarchical clustering. Data Skeleton Modeling (DSM) further refines GSOM’s interpretability by extracting skeletal structures from clusters. While these advancements are promising, empirical validation of GSOM’s topology preservation remains limited, and DSM integration has not been extensively compared to modern techniques like HDBSCAN or deep learning-based clustering methods. This research addresses these gaps by providing a comprehensive evaluation and extension of GSOM.

## Methodology
The research adopts a mixed-methods approach with three phases:

1. **Theoretical Validation:** Compare GSOM’s topology preservation with SOM using metrics like topographic error, topographic product, Zrehen measure, and C-measure.
2. **Experimental Evaluation:** Test GSOM’s hierarchical clustering on benchmark (UCI Zoo, Iris) and real-world datasets, comparing with K-Means, DBSCAN, GMM, and SOM using silhouette score, Davies-Bouldin index, and cluster purity.
3. **DSM Investigation:** Develop a framework to extract and visualize data skeletons, analyzing GSOM’s ability to model internal data structures and idionomic features in complex datasets.

## Experiment Setup and Implementation
* **Datasets:** Benchmark datasets from the UCI Repository (e.g., Zoo, Iris) and real-world datasets from Kaggle and open data portals.
* **Tools:** Python with libraries including NumPy, Pandas, SciPy, PYGSOM for GSOM implementation, Bigtree for hierarchical visualization, and Matplotlib/Seaborn for plotting.
* Implementation: Leverage existing DSM-enhanced GSOM code, with custom modules developed for data preprocessing, topology preservation evaluation, hierarchical representation, and visualization.
* Validation: Multiple datasets will be used, with experiments repeated to ensure reliability, and results compared against state-of-the-art clustering methods.
## Results and Analysis

## Conclusion

## Publications
[//]: # "Note: Uncomment each once you uploaded the files to the repository"

<!-- 1. [Semester 7 report](./) -->
<!-- 2. [Semester 7 slides](./) -->
<!-- 3. [Semester 8 report](./) -->
<!-- 4. [Semester 8 slides](./) -->
<!-- 5. Author 1, Author 2 and Author 3 "Research paper title" (2021). [PDF](./). -->


## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/repository-name)
- [Project Page](https://cepdnaclk.github.io/repository-name)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
