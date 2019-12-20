## Executive Summary & Introduction

The purpose of this analysis is to predict the number of cases of Dengue Fever in San Juan, Puerto Rico and Iquitos, Peru given environmental variables for a specific week of the year.  Because the transmission method of Dengue Fever is mosquitos, the transmission dynamics are related to environmental variables such as precipitation, humidity, and minimum & maximum temperatures.  In this paper, we will review the exploratory data analysis, data preparation, model building, and model selection processes that led us develop and build our predictive model. 
Dengue Fever is a mosquito-borne illness that originates in tropical and subtropical climates.  Contracting the virus can result in fever, rash, aches, and in the most severe cases death.  In the past, Dengue was most prevalent to Southeast Asia and Pacific Islands however, it has recently spread to Latin America with over a half-billion cases per year being reported.  Because the transmission rate of the illness is related to the mosquito population, many researchers believe there is a link between the spread of the virus and climate change.  By developing a better understanding of the environmental factors that contribute to the spread of Dengue Fever, Governments, Non-Governmental Institutions, and Researchers can develop policy and procedures to prevent the next Dengue Fever pandemic.

## Exploratory Data Analysis

The Dengue Fever data set consist of the response variable, total_cases, and 25 other potential regressor variables that indicate the city of occurrence (San Juan or Iquitos) and several environmental factors relating to vegetation growth, humidity, and temperature that span the period from 1990 to 2010.  There is a total of 2270 records of data and several of the regressor are missing values.

A plot of the regressor variable, total_cases, time series components shows potential pandemics of Dengue occurring roughly around the early 1990s, 1994, and 1997.  The period following the potential outbreak in 1997 to 2010 has been relatively stable with a moderate increase the number of cases of Dengue occurring 2005. There is clear seasonal cycle of total cases that peaks during the summer months, with warmer weather, and declines as the weather cools. This seasonal cycle adheres to our assumption that warmer, wetter, weather contributes to more mosquitos and cases of Dengue among the resident population.  The trend of total_cases does not have a consistent slope (rate of Dengue spread), rather there are several changes of directions that appear to be consistent with past outbreaks of Dengue Fever.  There may be several reasons for this change in trend ranging from implementation of specific policies to address the spread of mosquitos and Dengue Fever to simple fluctuations in weather patterns that created pools of water for mosquitos to breed and reproduce.  The random, or noise, portion of the decomposition appears to track with observed plot of total_cases which indicates that there is not a great deal of random noise in our total_cases time series.  

## Decomposition of Total_Case Time Series

Computing the Pearson Correlation Coefficients of our data set allow us to better understand the linear relationship between the response and regressor variables.  All the vegetation index variables (ndvi_ne, ndvi_nw, ndvi_se, ndvi_sw) appear to have a minimal inverse relationship with response variable total_cases.  This minimal relationship will lead us to remove this variables from our analysis as they initially appear to not significantly contribute to the number of casese of Denuge Fever.  Counter to intuition, the precipitation related regressors (precipitation_amt_mm, reanalysis_precip_amt_kg_per_m2, reanalysis_sat_precip_amt_mm,  station_precip_mm) have a minimal inverse relationship with total_cases.  This seems strange because large pools of water provide places for mosquitos to lay their eggs and reproduce, increasing the number of transmitters.  We will also remove these variable from our initial analysis because they appear to have minimal association with our response variable total_cases.  The regressor variables that relate to air temperature (reanalysis_air_temp_k, reanalysis_avg_temp_k, reanalysis_max_air_temp_k, reanalysis_min_air_temp_k, station_avg_temp_c, station_diur_temp_rng_c, station_max_temp_c , station_min_temp_c, reanalysis_tdtr_k) with an emphasis on the minimum temperatures had a relatively strong linear relationship with the response variable.  I’m hypothesizing that below a certain minimum temperature, the mosquito eggs and larvae die thereby reducing the overall population of Dengue Fever transmitters.  The predictor variables that relate to humidity (reanalysis_relative_humidity_percent,  reanalysis_specific_humidity_g_per_kg, reanalysis_relative_humidity_percent, reanalysis_air_temp_k, reanalysis_dew_point_temp_k) also appear to have relatively strong relationship with the response variable total_cases.  The computation and analysis of the Pearson Correlation Coefficient, has led me to believe that warmer, more humid temperatures contribute to the spread of Dengue Fever in San Juan, Puerto Rico and Iquitos, Peru.

## Data Preparation

Data preparation efforts consisted of variable transformation for ordinary least squares regression and imputation of missing records.  While not necessary for generalized linear models, to include logistic regression, Box Cox variable transformation produced an improved predictive model over other methods of transformation.  Logistic, additive, and multiplicative transformation were attempted, however the Box Cox transformation proved to be superior.  Adhering to the assumptions of linear regression (i.e. normal and independent distribution, linear relationship between variables, no or limited autocorrelation/multicollinearity, homoscedasticity) produced a more accurate model.  Rather than using measure of central tendency to replace missing records, I choose to use the last value in the regressors’ time series to replace any missing records.  This imputation methodology seemed the most logical because it reflected the tendencies of that variable, at that specific week in time. No outliers were removed from the data to their effects upon the selected Linear, Logistic, and Poisson Regression models.

## Model Development & Selection
	
During our analysis, we built multiple multivariate linear regression, Neural Network, and time series models utilizing Poisson, Negative Binomial, and ARIMA regression with both untransformed and transformed (box cox transformation) variables.  Following variable transformation and imputation, we progressed to splitting the designated training data 70/30 with 70% composing the revise training set and 30% creating a new validation data set.  This validated data was used to compare each model’s predictions against the validated data’s total_cases, using the mean absolute error (MAE) to assess goodness of fit.  Due to the number of variables in the provided data set, we selected our chosen model’s regressor variables based off the strength of their linear relationship with the response variable total_cases.   A summary of the accuracy outcomes is below for comparison:

#### Model Type  MAE
* General Linear Regression		15.53116
* Negative Binomial	        	12.4032
* Poisson                         	12.40738
* Neural Network w/ Log Xfer      	14.09859
* Gen. Linear Regression w/Box Cox Xfer 14.53373
* Negative Binomial w/Box Cox Xfer	12.46298
* Poisson w/Box Cox Xfer		12.48098
* Neural Network			15.14978
* ARIMA of Total_Cases			13.4056
* Neural Network w/ Log & Box Cox Xfer	19.40339
* Negative Binomial w/ Box Cox Xfer of reanalysis_min_air_temp	12.402

Our selected model, is a multivariate negative binomial linear regression model incorporating a box-cox transformation of the regressor variable reanalysis_min_air_temp that takes the form:

In Model
In Data
Beta
Value
Y
Total_Cases

-7.936e+01
X1
reanalysis_air_temp_k

5.510e-01
X2
reanalysis_avg_temp_k

-2.693e-01
X3
reanalysis_min_air_temp_k_BC

-1.008e-77
X4
reanalysis_tdtr_k

-4.913e-02
X5
station_diur_temp_rng_c

-4.678e-03
X6
station_min_temp_c

-4.372e-02

Error Term

## Model Notes:
* reanalysis_avg_temp_k has a negative coefficient which is counter to intuition.  I would hypothesize that warmer temperatures would contribute to more cases of Dengue by creating favorable conditions for mosquito eggs and larvae.

* reanalysis_min_air_temp_k underwent a box cox transformation that improved its R value from 0.1882959 to 0.188747.

## Summary

During this analysis, we built multiple multivariate linear regression, Neural Network, and time series models utilizing Poisson, Negative Binomial, and ARIMA regression with both untransformed and transformed (box cox transformation) variables to predict the total number of case of Dengue Fever for a given week in San Juan, Puerto Rico and Iquitos, Peru.  Exploratory Data Analysis was conducted to understand the distributions and relationships between our regressor and response variables.  We then implemented regressor variable Box Cox transformations to adhere to the assumptions of Linear Regression and improve the performance of our models.  Finally, an iterative trail-and-error process using Mean Absolute Error to assess goodness-of-fit was utilize to compare the results of each model to validation data and pick our selected model.  Despite our initial hypothesis that more rain would contribute to increase cases of Dengue we found that humidity and warmer temperatures were more significant to the spread of the virus.   If we had more time, we would like to further explore the performance of the negative binomial model over our other models, attempt additional variable transformations to further optimize our model, and implement dummy variables to incorporate the seasonality that we observed in the total_cases decomposed time series. 








