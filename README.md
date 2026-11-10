# 🚀 Data Science Mini Projects  

This repository contains multiple **data science projects** that demonstrate skills in **data cleaning, visualization, classification, clustering, and regression modeling**. Each project is implemented in Python with end-to-end workflows.  

---

## 📂 Projects  

### 🚗 Car Price Prediction  
> Note: the code for this project is not included in this repository yet.

- **Goal:** Predict used car prices based on features such as mileage, fuel type, and engine power.  
- **Techniques:** Data cleaning, missing value handling, outlier removal, regression modeling.  
- **Models Used:** Linear Regression, Random Forest Regressor.  
- **Outcome:** Built a predictive model to estimate car prices with high accuracy.  

---

### 🧬 Diabetes Prediction (Classification)  
- **Goal:** Predict whether a patient is diabetic or not using health parameters.  
- **Techniques:** Data preprocessing, distribution analysis, feature scaling.  
- **Models Used:** Logistic Regression, Random Forest, Support Vector Machine (SVM).  
- **Outcome:** Achieved strong classification accuracy for early diabetes detection.  

---

### 👥 Customer Segmentation (K-Means Clustering)  
- **Goal:** Segment customers into groups based on behavior for personalized strategies.  
- **Techniques:** Feature scaling, Elbow Method, clustering analysis.  
- **Algorithm Used:** K-Means Clustering.  
- **Outcome:** Identified distinct customer groups, enabling targeted marketing insights.  

---

### 📊 Exploratory Data Analysis (EDA)  
- **Goal:** Understand dataset structure, detect patterns, and discover insights before modeling.  
- **Techniques:**  
  - Correlation heatmaps  
  - Boxplots for outlier detection  
  - Distribution plots for feature analysis  
- **Outcome:** Extracted actionable insights and improved feature understanding.  

---

## 🛠️ Tech Stack  
- Python 🐍  
- Pandas, NumPy  
- Scikit-learn  
- Seaborn, Matplotlib  

---

---

## ▶️ Running the projects

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib

cd clustering && python cust.py            # customer segmentation (K-Means)
cd diabetes   && python diabetes_ml.py     # trains 3 classifiers, saves the best model
python diabetes_ml.py --predict new_rows.csv   # score new patients with the saved model
```
