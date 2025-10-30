# Multi-Class Model Selection using ROC and Precision-Recall Curves

## Assignment Overview

This Assignment addresses the DA5401 A7 assignment, focusing on the selection of a machine learning model for a multi-class classification task. The goal is to classify land cover types using the **UCI Landsat Satellite dataset**.

The core of the analysis moves beyond simple accuracy to a more nuanced evaluation using **Receiver Operating Characteristic (ROC)** curves and **Precision-Recall (PRC)** curves. By comparing a diverse set of classifiers—from simple baselines to powerful ensembles—the project aims to identify the most robust model and understand the performance trade-offs revealed by different evaluation metrics.

## Key Features

- **Comprehensive Model Comparison:** Evaluates 9 different classifiers, including KNN, Decision Tree, Logistic Regression, SVC, RandomForest, and XGBoost.
- **Advanced Evaluation Metrics:** Utilizes macro-averaged ROC-AUC and Precision-Recall Average Precision (PRC-AP) for robust multi-class model assessment.
- **Demonstration of "Worse-than-Random" performance:** Includes a custom classifier (`InvertedLogistic`) to illustrate the concept of an AUC score below 0.5.
- **Data Preprocessing:** Implements essential steps like feature standardization and label encoding.
- **Hyperparameter Tuning:** Uses `GridSearchCV` to find reasonable parameters for key models.
- **Detailed Analysis & Visualization:** The final Jupyter Notebook provides clear plots and in-depth text cells that tell a compelling data story, from initial exploration to the final recommendation.

## Dataset

The project uses the **Statlog (Landsat Satellite) Data Set** from the UCI Machine Learning Repository.
- **Citation:** Blake, C. and Merz, C.J. (1998). *UCI Repository of machine learning databases*. Irvine, CA: University of California, Department of Information and Computer Science.
- **Characteristics:**
  - **Features:** 36 integer attributes representing multi-spectral values of pixels in a 3x3 grid.
  - **Classes:** 6 distinct land cover types (classes 1, 2, 3, 4, 5, and 7).
  - **Instances:** The data is pre-split into a training set (`sat.trn`, 4435 instances) and a testing set (`sat.tst`, 2000 instances).

## How to Run the Project

### Prerequisites

- Python 3.x
- Jupyter Notebook or Google Colab
- Required Python libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`.

### Setup Instructions

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone https://github.com/killollik/DA5401_A7
    ```

2.  **Install Dependencies:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn xgboost
    ```

3.  **Obtain the Dataset:**
    - Download the data files from the [UCI Data Folder](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/).
    - You will need two files: `sat.trn` and `sat.tst`.
    - Place these two files in the root directory of the project.

4.  **Run the Jupyter Notebook:**
    - Launch Jupyter Notebook:
      ```bash
      jupyter notebook
      ```
    - Open the `DA5401_A7.ipynb` file.
    - Run the cells in order from top to bottom.

    **Alternatively, using Google Colab:**
    - Open [colab.research.google.com](https://colab.research.google.com).
    - Go to `File > Upload notebook` and select `DA5401_A7.ipynb`.
    - In the Colab file browser (left sidebar), click the "Upload" button and upload `sat.trn` and `sat.tst`.
    - Run the cells in order.

## Summary of Findings

The analysis produced a clear hierarchy of model performance, with key insights derived from comparing ROC and PRC curves.

### Model Performance Rankings

| Model              | Weighted F1-Score | ROC AUC | PRC Average Precision |
| ------------------ | ----------------- | ------- | --------------------- |
| **RandomForest**   | **0.910**         | 0.989   | **0.951**             |
| XGBoost            | 0.903             | **0.990** | **0.951**             |
| KNN                | 0.904             | 0.979   | 0.926                 |
| SVC                | 0.893             | 0.984   | 0.918                 |
| LogisticRegression | 0.830             | 0.975   | 0.871                 |
| ...                | ...               | ...     | ...                   |

### Key Insights

1.  **Top Tiers:** `RandomForest` and `XGBoost` were the undisputed top performers across all metrics, demonstrating both excellent general separability (high ROC-AUC) and robust performance on positive classes (high PRC-AP).

2.  **ROC vs. PRC Trade-off:** A notable trade-off was observed between `SVC` and `KNN`. While `SVC` achieved a higher ROC-AUC, `KNN` showed a superior PRC-AP. This suggests that if the application requires high precision (minimizing false positives), `KNN` would be the better choice, a subtlety that would be missed by only looking at ROC curves.

3.  **Worse-than-Random Model:** The custom `InvertedLogistic` classifier successfully achieved a ROC-AUC of **0.027**, perfectly illustrating how a model can perform systematically worse than random chance.

### Final Recommendation

The **Random Forest classifier** is recommended as the best overall model for this task.

**Justification:** It delivers top-tier performance that is statistically tied with XGBoost across all major metrics (F1-Score, ROC-AUC, PRC-AP). It is a robust, well-rounded model that excels at both general classification and high-precision predictions, making it the most reliable choice.
