Portfolio 1: Analysis of a car sell record Dataset
Overview
This Jupyter Notebook focuses on cleaning and preprocessing a dataset of car sales. The dataset contains multiple features including car ownership, fuel type, and seller type, among others. The main objective of the notebook is to remove outliers and filter the data based on specific rules. In this Read Me file is the overview of the portfolio and the main findings.

Dataset Description
The dataset used in this notebook contains the following key columns:

Owner: Describes the ownership type, such as 'First Owner', 'Second Owner', etc.
Fuel: Specifies the fuel type, such as 'Diesel' or 'Petrol'.
Seller Type: Indicates whether the seller is an 'Individual' or a 'Dealer'.
Other columns: Additional information about the cars, such as price, model, etc.
Notebook Structure
1. Data Loading
The dataset is loaded and the initial structure is examined to ensure it is in a usable format.

2. Outlier Removal
Outliers are removed in a series of steps based on the following rules:

Only cars with ownership of 'First Owner', 'Second Owner', or 'Third Owner' are retained.
Only 'Diesel' and 'Petrol' fuel types are kept.
Only sellers that are 'Individuals' or 'Dealers' are retained.
Each step prints the size of the dataset after applying the corresponding filter.

3. Final Cleaned Dataset
After all outliers are removed, the final cleaned dataset is displayed and is ready for further analysis or modeling.

Conclusion for Portfolio 1: Analysis of Car Sales Data
The portfolio effectively demonstrated the process of cleaning, preprocessing, and visualizing a car sales dataset. The key insights from the visualizations provide a deeper understanding of how various features impact the selling price of cars:

Year vs Selling Price: The analysis revealed that newer cars generally have higher selling prices, which aligns with the expected depreciation of cars over time. Older cars show a wider distribution in selling prices, indicating variability based on other factors such as condition and mileage.

Seller Type vs Selling Price: The plot comparing seller types showed that cars sold by dealers tend to have higher prices than those sold by individuals. This is likely due to added services such as warranties or repairs offered by dealers, making their cars more expensive.

Through the steps of data cleaning, including outlier removal and filtering by key attributes like fuel type and ownership, the dataset was refined for further analysis. These insights can be used for developing models to predict car prices or to make informed decisions in the used car market.

Usage
To run the notebook, open it in Jupyter Notebook or any compatible environment and execute the cells in sequence. The final output will be a cleaned dataset that has the following dimensions: `3657 rows and 8 co

Files Included
48622613_Portfolio1.ipynb - The Jupyter Notebook with all the code and analysis.
data/ - A folder containing the dataset car_sell.cvs lumns`.
Dependencies
Python 3.x
Pandas
Jupyter Notebook
Ensure all required packages are installed by running:

pip install pandas jupyter


;

Portfolio 2 - Car Sales Price Prediction
Project Overview
This Jupyter Notebook, titled "Portfolio 2 - Car Sales Price Prediction", is focused on predicting car selling prices using machine learning techniques, specifically linear regression models. The project explores the impact of feature selection and training data size on model performance. In addition to the predictive modeling, the notebook addresses ethical concerns in data science, particularly in data visualization and model transparency.

Introduction
The goal of the notebook is to train and evaluate different linear regression models for predicting the selling prices of cars. The notebook explores how feature selection and the size of training data affect the model’s accuracy and error metrics, along with discussing ethical considerations in data handling.

Libraries Used
pandas
numpy
matplotlib
scikit-learn
Data Loading and Preprocessing
The dataset used in this project is a cleaned version of a car sales dataset containing columns such as year, selling price, kilometers driven, fuel type, seller type, transmission, and number of previous owners. Categorical variables are converted to numerical values to be used in the regression models.

Analysis / Modeling
Four linear regression models are trained to predict selling prices:

Model A: Uses two most correlated features (year, transmission) with 10% training data.
Model B: Uses two least correlated features (kilometers driven, owner) with 10% training data.
Model C: Uses two most correlated features with 90% training data.
Model D: Uses two least correlated features with 90% training data.
The models are compared using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score to assess their performance.

Data Science Ethics
This section addresses potential ethical concerns in data visualization and model transparency. The notebook includes an analysis of ethical issues, referencing two key resources:

Example 1: A case from Georgia where COVID-19 data visualization led to misinterpretation of trends.
Example 2: An article on ethical data visualization, emphasizing the importance of context and transparency in presenting data.
Ethical issues discussed include:

Misrepresentation of data through improper visualization techniques.
The importance of clearly labeling axes and providing sufficient context to avoid misleading interpretations.
Transparency in machine learning models to ensure that predictions are not only accurate but also understandable and explainable.
Results
Model C (90% training data and most correlated features) had the best performance with the lowest MSE and RMSE.
Model B (10% training data and least correlated features) performed the worst, highlighting the importance of both feature selection and data size.
Conclusion
Feature relevance and training data size significantly affect the accuracy of the regression models. The Year feature had the highest positive correlation with selling price, suggesting that newer cars command higher prices.

Future Work / Improvements
Additional features could be explored to improve the model’s predictive power, as the R² values indicate that current features do not fully explain the variability in selling prices.

Files Included
48622613_Portfolio2.ipynb - The Jupyter Notebook with all the code and analysis.
data/ - A folder containing the dataset loan_approval.cvs
How to Run the Notebook
To run the notebook, first install the necessary packages:

pip install pandas jupyter
