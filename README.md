# HEART DISEASE PREDICTION
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/aaeb377b7ed623d05626cabf101caf74542bb88a/Blog/heart_title.jpg)

## PROJECT ABSTRCT
Predicting heart disease has become a critical focus of healthcare research due to its high prevalence and significant impact on public health. Early detection of heart disease can significantly reduce the risk of severe outcomes by enabling timely intervention and personalized treatment plans. This project aims to develop a predictive model for heart disease using machine learning techniques. By analyzing a dataset of patient health records, which includes various features such as age, gender, blood pressure, cholesterol levels, and lifestyle factors, the model will predict the likelihood of a patient developing heart disease. The project will utilize various algorithms, such as logistic regression, decision trees, and random forests, to identify patterns and risk factors associated with heart disease. The goal is to create a user-friendly tool that healthcare professionals can use to make more informed decisions regarding patient care. Additionally, this predictive model seeks to improve patient outcomes through early intervention and personalized risk management strategies.

## ABOUT DATASET
This dataset gives information related to heart disease. The dataset contains 13 columns, target is the class variable which is affected by the other 12 columns. Here the aim is to classify the target variable to (disease\non disease) using different machine learning algorithms and find out which algorithm is suitable for this dataset.
## HOW DATASET GENERATED
The make_classification function in Python, part of the sklearn.datasets module, is commonly used to simulate classification datasets for testing machine learning algorithms. It allows you to generate synthetic datasets with a specified number of features, classes, informative and redundant features, and noise. This is particularly useful when you want to experiment with classification models and benchmark their performance using controlled, generated data.

Here’s how you can simulate data using the make_classification function:
```bash
from sklearn.datasets import make_classification
```
## ATTRIBUTES
Age\
Gender\
Chest Pain Type\
Resting Blood Pressure\
Serum Cholesterol\
Fasting Blood Sugar\
Resting Electrocardiographic Results\
Maximum Heart Rate Achieved\
Exercise-induced angina\
Depression induced by exercise relative to rest\
Slope of the Peak Exercise ST Segment\
Number of Major Vessels Colored by Fluoroscopy\
Thalassemia\
Target

## BUILT WITH
NUMPY\
PANDAS\
MATPLOTLIB\
SEABORN\
SCIKITLEARN\
VISUAL STUDIO CODE\
STREAMLIT\
JOBLIB

## HOW DATA LOOK LIKE AFTER GENERATION
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/dataset.jpg)

## DATA INFORMATION
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/data_info.jpg)

## COUNTPLOT OF LABEL(TARGET)
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/count_plot.jpg)

## DISTRIBUTION PLOT
A distribution plot is a visual tool used in machine learning and data analysis to understand how a dataset is distributed, which can provide valuable insights for preprocessing and model selection. It displays the frequency or density of different values within a feature or variable, helping to identify patterns, outliers, and the overall structure of the data. Below are some of the key aspects of distribution plots in machine learning.
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/distributions_of_colums.png)

## BOX PLOT
A boxplot (also known as a box-and-whisker plot) is a graphical representation used to summarize the distribution of a dataset, showing its central tendency, spread, and identifying potential outliers. Boxplots are widely used in exploratory data analysis (EDA) to quickly get a sense of the data distribution and detect any abnormalities.
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/boxplots.png)

## HEATMAP
A heatmap is a data visualization technique that represents data in a matrix format, where individual values are represented by color. Heatmaps are useful for visualizing complex data sets, especially when looking for patterns, correlations, or trends. In machine learning, heatmaps are often used to show correlation matrices, feature importance, or the relationship between variables.
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/heatmap.png)

## ZSCORE
The Z-score, also known as the standard score or normal deviate, is a statistical measure that describes how far a particular data point is from the mean of the dataset in terms of standard deviations. In the context of machine learning, the Z-score is often used for data standardization, a crucial preprocessing step in many algorithms, especially those that are sensitive to the scale of the data, such as linear regression, support vector machines (SVMs), and neural networks.
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/zscore.jpg)

## VARIATION INFLATION FACTOR(VIF)
The Variance Inflation Factor (VIF) is a measure used to detect multicollinearity in regression models. Multicollinearity occurs when two or more independent variables (features) in a model are highly correlated with each other, which can lead to unreliable estimates of the coefficients and reduce the interpretability of the model.
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/vif.jpg)

## MODELS USED TO BUILT MODEL
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/ml_models_names.jpg)

## CODE TO GET BEST RANDOMSTATE THAT GIVES LOW ERROR RATE
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/best_random_state.jpg)

## GET BEST NEIGHBOUR FOR KNN
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/best_k_rmse.jpg)

## PLOT FOR GETTING BEST NEIGHBOUR IN KNN
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/best_k_for_knn.png)

## RUNNING ALGORITHMS IN LOOP TO GET RESULTS
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/code_for_all_ml_algorithms_loop.jpg)

## REPORT OF RANDOMFOREST CLASSIFIER
A Random Forest Classifier is an ensemble learning method used for classification tasks. It builds multiple decision trees during training and combines their outputs to make more accurate predictions. It is one of the most popular machine learning algorithms due to its high performance, ability to handle complex datasets, and resilience to overfitting.
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/random_classifier_report.jpg)

## CROSS VALIDATION
Cross-validation is a model validation technique used to assess the performance of a machine learning model on unseen data. It helps in evaluating how well the model generalizes to an independent dataset. The main goal of cross-validation is to ensure that the model's performance is robust and not overly optimistic or pessimistic due to overfitting or underfitting.
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/crossval_score.jpg)

## WHY TO RANDOMFOREST IS BETER FOR THIS PROBLEM
Key Disadvantages:
#### Complexity:
Random Forest models can be computationally expensive and take up a lot of memory when dealing with large datasets or many trees.
#### retability:
While decision trees are interpretable, Random Forests, which aggregate many trees, are harder to interpret compared to individual decision trees.
#### Slow for Predictions:
Since predictions require passing the input through many trees, the process can be slower compared to simpler models (e.g., logistic regression or k-nearest neighbors) during inference.

## GRIDSEARCH
Grid Search is a hyperparameter tuning technique used to find the best combination of hyperparameters for a machine learning model. It systematically searches through a manually specified subset of the hyperparameter space to determine the optimal set of hyperparameters that yields the best model performance.
Grid search is most commonly used in conjunction with cross-validation to assess how different combinations of hyperparameters affect the performance of the model on unseen data.
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/grid_search_code.jpg)
#### BEST PARAMETERS FROM GRIDSEARCH
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/grid_best_params.jpg)

## AREA UNDER THE CURVE(AUC)
AUC stands for Area Under the Curve and is often used as a performance metric for classification models, especially in binary classification tasks. It is typically used in the context of the Receiver Operating Characteristic (ROC) curve and is referred to as AUC-ROC.
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/auc_curve.png)

## JOBLIB
Joblib is a Python library used for serialization (saving) and deserialization (loading) of Python objects. It is especially useful for saving machine learning models, large numpy arrays, and any objects that are time-consuming to recompute. Joblib provides an efficient way to persist and reload objects, which can be extremely helpful when deploying machine learning models or saving intermediate results in data analysis workflows.
Joblib is an alternative to the built-in pickle module, and it is optimized for handling large data, such as NumPy arrays or machine learning models, that might otherwise consume a lot of memory and take longer to serialize and deserialize with pickle.
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/joblib.jpg)

## NOTE
After crration of joblib file import that file into visual studio code and import necessary liabraries of streamlit and write code to excute it.

## SETTING VISUAL STUDIO CODE FOR STREAMLIT
To set up Visual Studio Code (VS Code) for Streamlit development, follow these steps to install and configure everything you need to run and develop Streamlit apps efficiently.
#### 1. Install Visual Studio Code
If you don’t have Visual Studio Code installed already, follow these steps:
Download and install VS Code from the official site: https://code.visualstudio.com/
Follow the installation instructions for your operating system (Windows, macOS, or Linux).
#### 2. Install Python and Streamlit
You need to have Python installed on your system to use Streamlit. If Python is not installed, follow these steps:
Install Python:
Download Python from https://www.python.org/downloads/.
Install Python and ensure you select the option to add Python to the PATH during installation.
Install Streamlit: Once Python is installed, open a terminal (or the terminal inside VS Code) and install Streamlit using pip:
```bash
pip install streamlit
```
#### 3.Create a Streamlit App
Inside your project folder, create a Python file (e.g., app.py).
Write a simple Streamlit app in that file. Here’s a basic example:
Install Streamlit: Once Python is installed, open a terminal (or the terminal inside VS Code) and install Streamlit using pip:
```bash
import streamlit as st

st.title('Hello, Streamlit!')
st.write('Welcome to your first Streamlit app.')

if st.button('Say Hello'):
    st.write('Hello World!')
```
#### 4.Run Streamlit App in VS Code
Open the terminal in VS Code (Ctrl+`` or Cmd+``) and navigate to the folder where your app.py is located (if not already there).
Run the Streamlit app using the following command:
```bash
streamlit run app.py
```

## VISUALIZATION AND PERFORMANCE OF MACHINE LEARNING ALGORITHMS USING STREAMLIT
Streamlit is an excellent framework for creating interactive data visualizations and web applications with Python. It allows you to easily build dashboards, plots, and interactive widgets without requiring extensive web development skills.
Here’s a simple guide on how to create a basic visualization using Streamlit. I'll walk you through a common example of plotting a graph (e.g., a line plot) with interactivity using libraries such as matplotlib or plotly.
### LIBRARIES FOR CREATION OF VISUALIZATION OF MACHINE LEARNING ALGORITHMS USING STREAMLIT
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/aaeb377b7ed623d05626cabf101caf74542bb88a/Blog/Visualization_part_1.jpg)
### LIST OF MACHINE LEARNING ALGORITHMS USED IN MODEL BUILDING AND CHECK THEIR PERFORMANCE THROUGH STREAMLIT VISUALIZATION
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/aaeb377b7ed623d05626cabf101caf74542bb88a/Blog/Visualization_options.jpg)
#### K-NEAREST NEIGHBOUR (KNN CLASSIFIER)
KNN is a supervised learning algorithm, meaning it requires labeled data (i.e., data with both input features and known output labels) to make predictions.
The basic idea of KNN is to classify a data point by looking at the "K" nearest labeled data points in the feature space and taking a majority vote (for classification) or averaging the values (for regression).
##### KNN PERFORMANCE
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/aaeb377b7ed623d05626cabf101caf74542bb88a/Blog/Visualization_knn_part_1.jpg)
##### KNN VISUALIZATION
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/aaeb377b7ed623d05626cabf101caf74542bb88a/Blog/Visualization_knn_part_2.jpg)
#### LOGISTIC REGRESSION
Logistic Regression is a powerful algorithm used for binary classification tasks. Despite its name, it is a classification algorithm rather than a regression algorithm. Logistic regression models the probability of a binary outcome (e.g., success/failure, 0/1, true/false) based on one or more input features.
##### LOGISTIC REGRESSION PERFORMANCE
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/aaeb377b7ed623d05626cabf101caf74542bb88a/Blog/Visualization_logistic_part_1.jpg)
##### LOGISTIC REGRESSION VISUALIZATION
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/aaeb377b7ed623d05626cabf101caf74542bb88a/Blog/Visualization_logistic_part_2.jpg)
#### SUPPORT VECTOR MACHINE(CLASSIFIER)
A Support Vector Classifier (SVC) is a type of supervised machine learning algorithm that is used for classification tasks. It is part of the family of Support Vector Machines (SVMs), which are particularly effective for classification problems where the decision boundary is not necessarily linear. The key idea behind SVC is to find the optimal hyperplane that best separates the classes in the feature space, maximizing the margin between the classes.
##### SUPPORT VECTOR MACHINE(CLASSIFIER) PERFORMANCE
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/aaeb377b7ed623d05626cabf101caf74542bb88a/Blog/Visualization_svm_part_1.jpg)
##### SUPPORT VECTOR MACHINE(CLASSIFIER) VISUALIZATION
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/aaeb377b7ed623d05626cabf101caf74542bb88a/Blog/Visualization_svm_part_2.jpg)
#### DECISIONTREE CLASSIFIER
A Decision Tree Classifier is a popular machine learning algorithm used for both classification and regression tasks. It is a non-parametric method that works by splitting the data into subsets based on feature values, recursively forming a tree-like structure. The algorithm divides the feature space into regions that are as homogeneous as possible with respect to the target variable (class label for classification tasks).
##### DECISIONTREE CLASSIFIER PERFORMANCE
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/aaeb377b7ed623d05626cabf101caf74542bb88a/Blog/Visualization_decisiontree_part_1.jpg)
##### DECISIONTREE CLASSIFIER VISUALIZATION
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/aaeb377b7ed623d05626cabf101caf74542bb88a/Blog/Visualization_decisiontree_part_2.jpg)
#### GRADIENTBOOSTING CLASSIFIER
Gradient Boosting is a powerful and popular machine learning technique for both regression and classification tasks. It belongs to the family of ensemble learning methods, where multiple models are combined to make a more accurate prediction. Specifically, Gradient Boosting builds an ensemble of weak learners (typically decision trees) and combines them sequentially, where each new tree attempts to correct the errors made by the previous ones.
Gradient Boosting is highly effective, especially when dealing with complex datasets and provides competitive results in many machine learning competitions.
##### GRADIENTBOOSTING CLASSIFIER PERFORMANCE
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/aaeb377b7ed623d05626cabf101caf74542bb88a/Blog/Visualization_gradientboosting_part_1.jpg)
##### GRADIENTBOOSTING CLASSIFIER VISUALIZATION
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/aaeb377b7ed623d05626cabf101caf74542bb88a/Blog/Visualization_gradientboosting_part_2.jpg)
#### RANDOMFOREST CLASSIFIER
The Random Forest Classifier is an ensemble learning algorithm that uses multiple decision trees to make predictions. It is an extension of the bagging method and is known for its robustness, high accuracy, and ability to handle overfitting. It works by constructing multiple decision trees during training and combining their outputs to improve the final prediction.
##### RANDOMFOREST CLASSIFIER PERFORMANCE
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/aaeb377b7ed623d05626cabf101caf74542bb88a/Blog/Visualization_random_part_1.jpg)
##### RANDOMFOREST CLASSIFIER VISUALIZATION
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/aaeb377b7ed623d05626cabf101caf74542bb88a/Blog/Visualization_random_part_2.jpg)

## CODE TO CREATE STREAMLIT APP
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/vs_streamlit_code_part1.jpg)
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/vs_streamlit_code_part2.jpg)
### RUN CODE IN TERMINAL
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/streamlit_terminal.jpg)
### FEATURES(COLUMNS) NEED TO SELECT FROM WEBAPP AND CLICK ON PREDICTION TO GET RESULT
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/streamlit_colums_part_1.jpg)
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/streamlit_colums_part_2.jpg)
### HOW RESULT APPEAR IF PERSON HAS HEART DISEASE
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/heart_disease_yes.jpg)
### HOW RESULT APPEAR IF PERSON HAS NO HEART DISEASE
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e4b1c14fcd6c5f9fac53ea9cb506f8c41759958a/Blog/heart_disease_no.jpg)

## SUMMARY OF KEY FINDINGS:
Machine learning provides a powerful toolkit for predicting heart disease by analyzing patient data.
Proper data preprocessing, feature selection, and model evaluation are critical for building effective models.
Algorithms such as Random Forests, Logistic Regression, and SVM show promising results, but the choice of algorithm depends on the problem at hand and the importance of interpretability.
Predictive models can aid healthcare professionals in identifying at-risk patients early, improving diagnosis, and providing better treatment outcomes.
The focus on recall and AUC over raw accuracy ensures that the model minimizes the risk of missing heart disease diagnoses.
Ongoing research and advancements in AI and healthcare integration can lead to even more accurate and practical solutions for heart disease prediction.
#### In conclusion, machine learning holds significant promise in transforming healthcare by providing tools for early prediction, personalized treatment, and better management of heart disease.
![alt text](https://github.com/Vijaychilukuri/MachineLearningProjects/blob/e3f5afd9aaf427d3d4f06bf544cea38d0e5c1009/Blog/Robot-Thank-You.jpg)











































