M5 Sales Forecasting with Machine Learning

Overview

This project builds machine learning models to predict daily retail sales using the M5 Forecasting dataset. The objective is to compare multiple regression models and improve prediction accuracy through feature engineering and hyperparameter tuning.

Dataset

The project uses the M5 Forecasting dataset, which contains historical Walmart sales data.

Files used:
	•	sales_train_validation.csv – historical sales
	•	calendar.csv – calendar information and events
	•	sell_prices.csv – product prices over time

Key identifiers include item_id, dept_id, cat_id, store_id, and state_id. The target variable is sales.

Data Processing

The following preprocessing and feature engineering steps were applied:
	•	Merged sales data with calendar and price information
	•	Converted the dataset to a long format using melt
	•	Created time-based features: year, month, week, dayofweek, quarter, is_weekend
	•	Generated lag features: lag_7, lag_14, lag_28
	•	Computed rolling statistics: rolling_mean_7, rolling_mean_30, rolling_std_7
	•	Added aggregated features: store_avg_sales, dept_avg_sales
	•	Encoded categorical variables using LabelEncoder
	•	Filled missing numeric values with 0

Models

The following regression models were trained and compared:
	•	Linear Regression
	•	Random Forest Regressor
	•	Gradient Boosting Regressor
	•	XGBoost Regressor
	•	LightGBM Regressor

All models were implemented using a scikit-learn pipeline that includes preprocessing.

Hyperparameter Tuning

The two best-performing models were further optimized using RandomizedSearchCV:
	•	LightGBM
	•	Gradient Boosting

Evaluation

Models were evaluated using:
	•	Mean Absolute Error (MAE)
	•	Root Mean Squared Error (RMSE)

Final results on the test set:

Model Performance

Baseline Model Comparison

Model	                MAE	                                 RMSE
LightGBM	            0.9876                             	2.7358
Gradient Boosting    	0.9983	                            2.7452
Random Forest        	0.9969	                            2.7705
Linear Regression	    1.0231                            	2.7787
XGBoost              	0.9884                            	2.8738

Tuned Model Performance

Model	                             MAE
Tuned LightGBM	                   0.806

LightGBM achieved the best overall performance, reducing the MAE significantly compared to the baseline models.

Technologies:
Python, Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, Matplotlib


