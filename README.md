# Portfolio 4: Predicting Obesity Levels Based on Eating Habits and Physical Condition

## Overview

This Jupyter Notebook aims to predict obesity levels by analyzing various factors such as eating habits, physical condition, and personal attributes. It leverages two classification models: K-Nearest Neighbors (KNN) and Logistic Regression. The goal is to identify which model performs better in classifying instances of obesity and to interpret the significant features contributing to these predictions.

## Dataset

The dataset used is **"Estimation of Obesity Levels Based on Eating Habits and Physical Condition"**, sourced from the UCI Machine Learning Repository. It includes features related to:
- Age, gender, and personal habits
- Eating frequency (e.g., vegetable and snack consumption)
- Physical activity frequency
- Other personal metrics, such as family history of overweight or smoking habits

The dataset can be found [here](https://doi.org/10.24432/C5H31Z).

## Models and Techniques

### 1. **K-Nearest Neighbors (KNN)**
   - KNN is used as a baseline model to classify individuals into obesity categories.
   - The optimal value of K is determined through cross-validation, with K = 4 yielding the highest accuracy of 92%.

### 2. **Logistic Regression**
   - Recursive Feature Elimination (RFE) is applied to select the most important features contributing to obesity prediction.
   - The logistic regression model is evaluated based on accuracy and F1 score, and the identified features provide insights into obesity risk factors.

### 3. **Model Evaluation**
   - Both models are evaluated using accuracy, precision, recall, and F1 score.
   - Confusion matrices are generated to assess the classification performance for different obesity levels.

## Notebook Structure

### 1. Data Preprocessing
The dataset is loaded and cleaned, with necessary transformations and encodings applied to prepare it for model training.

### 2. KNN and Logistic Regression Training
The notebook trains both KNN and Logistic Regression models on the dataset, using cross-validation to determine the best-performing hyperparameters.

### 3. Feature Importance Analysis
The RFE method is applied to the Logistic Regression model to identify the most important features related to obesity. These include age, snack frequency, physical activity, and family history of overweight.

### 4. Results and Conclusion
- **KNN** achieved the highest accuracy (92%) and F1 score (0.89), making it the most effective model for this task.
- **Logistic Regression** also performed well but was slightly outperformed by KNN.
- Key features contributing to the predictions are discussed in detail, providing insights into obesity risk factors.

## Conclusions

- **KNN Model**: Achieved better accuracy and F1 score than Logistic Regression, making it the preferred model for obesity prediction in this dataset.
- **Logistic Regression**: Identified important features contributing to obesity, including frequency of snacks, vegetable consumption, family history of overweight, and physical activity frequency.
- **Optimal K for KNN**: K = 4 was found to be the optimal number of neighbors, yielding the best model performance.
- 
## Files Included

- `48622613_Portfolio4.ipynb` - The Jupyter Notebook with all the code and analysis.
- `data/` - A folder containing the dataset ObesityDataSet_raw_and_data_sinthetic.csv
- `Reflective Report.pdf` - File containing Final Reflective Report

## Usage

To explore the analysis, open the notebook in Jupyter Notebook or a compatible environment and execute the cells in sequence. The results will include model performance metrics, confusion matrices, and an analysis of important features.

## Dependencies

- Python 3.x
- Pandas
- Scikit-learn
- Jupyter Notebook


# Portfolio 3: KNN Classifier with Distance Metrics Evaluation

## Overview

This Jupyter Notebook evaluates the performance of a K-Nearest Neighbors (KNN) classifier using three different distance metrics:
- Euclidean Distance
- Cosine Distance
- Manhattan Distance

The goal is to determine which distance metric is the most effective in terms of accuracy and F1 score for the given dataset.

## Dataset

The dataset used in this notebook contains features and labels for classification. The exact features and labels are explored during the analysis, with a focus on how the relationships between data points influence the model's performance.

## Distance Metrics

### 1. **Euclidean Distance**:
   - Measures the straight-line distance between two points in space.
   - Achieved the highest accuracy and F1 score, making it the best metric for this dataset.
   
### 2. **Cosine Distance**:
   - Evaluates the cosine of the angle between two vectors.
   - Achieved second-best performance, close to Euclidean, indicating that the direction of data points has some significance.

### 3. **Manhattan Distance**:
   - Measures the distance between two points by summing the absolute differences of their coordinates.
   - Showed the lowest performance in both accuracy and F1 score.

## Notebook Structure

### 1. Data Preprocessing
The dataset is loaded and preprocessed to ensure it is in a format suitable for KNN classification.

### 2. KNN Model Training and Evaluation
The KNN classifier is trained using each of the three distance metrics (Euclidean, Cosine, and Manhattan). The model's accuracy and F1 score are calculated and compared.

### 3. Results and Conclusions
The performance of each distance metric is evaluated based on the model's accuracy and F1 score. The results show that Euclidean distance is the most effective metric for this dataset, followed closely by Cosine distance, while Manhattan distance shows limited effectiveness.

## Conclusions

1. **Euclidean Distance**: Achieved the highest accuracy and F1 score, making it the most effective metric for this dataset.
2. **Cosine Distance**: Performance was slightly lower than Euclidean but still close, indicating that the orientation of data points matters.
3. **Manhattan Distance**: Recorded the lowest F1 score, suggesting it is less effective in capturing the relationships between data points for this particular dataset.

## Files Included

- `48622613_Portfolio3.ipynb` - The Jupyter Notebook with all the code and analysis.
- `data/` - A folder containing the dataset loan_approval.cvs

## Usage

To run the notebook, simply open it in Jupyter Notebook or any compatible environment and execute the cells in order. The results of the analysis will show how each distance metric performs with the KNN classifier.

## Dependencies

- Python 3.x
- Pandas
- Scikit-learn
- Jupyter Notebook


# Portfolio 2 - Car Sales Price Prediction

## Project Overview

This Jupyter Notebook, titled **"Portfolio 2 - Car Sales Price Prediction"**, is focused on predicting car selling prices using machine learning techniques, specifically linear regression models. The project explores the impact of feature selection and training data size on model performance. In addition to the predictive modeling, the notebook addresses ethical concerns in data science, particularly in data visualization and model transparency.


## Introduction

The goal of the notebook is to train and evaluate different linear regression models for predicting the selling prices of cars. The notebook explores how feature selection and the size of training data affect the model’s accuracy and error metrics, along with discussing ethical considerations in data handling.

## Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

## Data Loading and Preprocessing

The dataset used in this project is a cleaned version of a car sales dataset containing columns such as year, selling price, kilometers driven, fuel type, seller type, transmission, and number of previous owners. Categorical variables are converted to numerical values to be used in the regression models.

## Analysis / Modeling

Four linear regression models are trained to predict selling prices:

- **Model A:** Uses two most correlated features (year, transmission) with 10% training data.
- **Model B:** Uses two least correlated features (kilometers driven, owner) with 10% training data.
- **Model C:** Uses two most correlated features with 90% training data.
- **Model D:** Uses two least correlated features with 90% training data.

The models are compared using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score to assess their performance.

## Data Science Ethics

This section addresses potential ethical concerns in data visualization and model transparency. The notebook includes an analysis of ethical issues, referencing two key resources:

1. **Example 1:** A case from Georgia where COVID-19 data visualization led to misinterpretation of trends.
2. **Example 2:** An article on ethical data visualization, emphasizing the importance of context and transparency in presenting data.

**Ethical issues discussed include:**

- Misrepresentation of data through improper visualization techniques.
- The importance of clearly labeling axes and providing sufficient context to avoid misleading interpretations.
- Transparency in machine learning models to ensure that predictions are not only accurate but also understandable and explainable.

## Results

- **Model C** (90% training data and most correlated features) had the best performance with the lowest MSE and RMSE.
- **Model B** (10% training data and least correlated features) performed the worst, highlighting the importance of both feature selection and data size.

## Conclusion

Feature relevance and training data size significantly affect the accuracy of the regression models. The **Year** feature had the highest positive correlation with selling price, suggesting that newer cars command higher prices.

## Future Work / Improvements

Additional features could be explored to improve the model’s predictive power, as the R² values indicate that current features do not fully explain the variability in selling prices.

## Files Included

- `48622613_Portfolio2.ipynb` - The Jupyter Notebook with all the code and analysis.
- `data/` - A folder containing the dataset loan_approval.cvs


# Portfolio 1: Analysis of a car sell record Dataset

## Overview

This Jupyter Notebook focuses on cleaning and preprocessing a dataset of car sales. The dataset contains multiple features including car ownership, fuel type, and seller type, among others. The main objective of the notebook is to remove outliers and filter the data based on specific rules.
In this Read Me file is the overview of the portfolio and the main findings.

## Dataset Description

The dataset used in this notebook contains the following key columns:

1. **Owner**: Describes the ownership type, such as 'First Owner', 'Second Owner', etc.
2. **Fuel**: Specifies the fuel type, such as 'Diesel' or 'Petrol'.
3. **Seller Type**: Indicates whether the seller is an 'Individual' or a 'Dealer'.
4. **Other columns**: Additional information about the cars, such as price, model, etc.

## Notebook Structure

### 1. Data Loading
The dataset is loaded and the initial structure is examined to ensure it is in a usable format.

### 2. Outlier Removal
Outliers are removed in a series of steps based on the following rules:
- Only cars with ownership of 'First Owner', 'Second Owner', or 'Third Owner' are retained.
- Only 'Diesel' and 'Petrol' fuel types are kept.
- Only sellers that are 'Individuals' or 'Dealers' are retained.

Each step prints the size of the dataset after applying the corresponding filter.

### 3. Final Cleaned Dataset
After all outliers are removed, the final cleaned dataset is displayed and is ready for further analysis or modeling.


## Conclusion for Portfolio 1: Analysis of Car Sales Data

The portfolio effectively demonstrated the process of cleaning, preprocessing, and visualizing a car sales dataset. The key insights from the visualizations provide a deeper understanding of how various features impact the selling price of cars:

1. **Year vs Selling Price**: The analysis revealed that newer cars generally have higher selling prices, which aligns with the expected depreciation of cars over time. Older cars show a wider distribution in selling prices, indicating variability based on other factors such as condition and mileage.

2. **Seller Type vs Selling Price**: The plot comparing seller types showed that cars sold by dealers tend to have higher prices than those sold by individuals. This is likely due to added services such as warranties or repairs offered by dealers, making their cars more expensive.

Through the steps of data cleaning, including outlier removal and filtering by key attributes like fuel type and ownership, the dataset was refined for further analysis. These insights can be used for developing models to predict car prices or to make informed decisions in the used car market.

## Usage

To run the notebook, open it in Jupyter Notebook or any compatible environment and execute the cells in sequence. The final output will be a cleaned dataset that has the following dimensions: `3657 rows and 8 co
## Files Included

- `48622613_Portfolio1.ipynb` - The Jupyter Notebook with all the code and analysis.
- `data/` - A folder containing the dataset car_sell.cvs
lumns`.

## Dependencies

- Python 3.x
- Pandas
- Jupyter Notebook

Ensure all required packages are installed by running:

```bash
pip install pandas jupyter


;

