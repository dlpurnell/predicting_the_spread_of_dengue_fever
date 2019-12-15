# Daren Purnell/Predict413/Section55/Final/DengAI

# Load Libraries
pkgs <- c('tidyverse', 'corrplot', 'magrittr', 'zoo', 'RColorBrewer', 'gridExtra','MASS', 
          'forecast', 'car', 'TTR', 'MVN', 'boot','ResourceSelection', 'leaps', 'pROC',
          'ggplot2', 'reshape2', 'caret', 'TSA', 'psych')
invisible(lapply(pkgs, require, character.only = T))

# DATA PREPARATION

# Data Loading
train_features = read.csv(file.choose(), header=TRUE, stringsAsFactors = F)
train_labels   = read.csv(file.choose(), header=TRUE, stringsAsFactors = F)

# Verify that data is loaded
head(train_features[1:7])
head(train_labels)

# Join the two sets of training data
training_data <- total <- merge(train_labels,train_features,by=c("year","weekofyear"))
summary(training_data)

# Check for missing values and NAs in our two training data sets
sapply(training_data,function(x) sum(is.na(x)))
sapply(training_data, function(x) length(unique(x)))

# Impute missing values with the last value
training_data%<>% na.locf(fromLast = TRUE)

# Convert to numeric values
training_data$total_cases <- as.numeric(training_data$total_cases)
training_data$year <- as.numeric(training_data$year)
training_data$weekofyear <- as.numeric(training_data$weekofyear)
training_data$week_start_date <- as.Date(training_data$week_start_date)
training_data$ndvi_ne <- as.numeric(training_data$ndvi_ne)
training_data$ndvi_nw <- as.numeric(training_data$ndvi_nw)
training_data$ndvi_se <- as.numeric(training_data$ndvi_se)
training_data$ndvi_sw <- as.numeric(training_data$ndvi_sw)
training_data$precipitation_amt_mm <- as.numeric(training_data$precipitation_amt_mm)
training_data$reanalysis_air_temp_k <- as.numeric(training_data$reanalysis_air_temp_k)
training_data$reanalysis_avg_temp_k <- as.numeric(training_data$reanalysis_avg_temp_k)
training_data$reanalysis_dew_point_temp_k <- as.numeric(training_data$reanalysis_dew_point_temp_k)
training_data$reanalysis_max_air_temp_k <- as.numeric(training_data$reanalysis_max_air_temp_k)
training_data$reanalysis_min_air_temp_k <- as.numeric(training_data$reanalysis_min_air_temp_k)
training_data$reanalysis_precip_amt_kg_per_m2 <- as.numeric(training_data$reanalysis_precip_amt_kg_per_m2)
training_data$reanalysis_relative_humidity_percent <- as.numeric(training_data$reanalysis_relative_humidity_percent)
training_data$reanalysis_sat_precip_amt_mm <- as.numeric(training_data$reanalysis_sat_precip_amt_mm)
training_data$reanalysis_specific_humidity_g_per_kg <- as.numeric(training_data$reanalysis_specific_humidity_g_per_kg)
training_data$reanalysis_tdtr_k <- as.numeric(training_data$reanalysis_tdtr_k)
training_data$station_avg_temp_c <- as.numeric(training_data$station_avg_temp_c)
training_data$station_diur_temp_rng_c <- as.numeric(training_data$station_diur_temp_rng_c)
training_data$station_max_temp_c <- as.numeric(training_data$station_max_temp_c)
training_data$station_min_temp_c <- as.numeric(training_data$station_min_temp_c)
training_data$station_precip_mm <- as.numeric(training_data$station_precip_mm)

# EXPLORATORY DATA ANALYSIS
summary(training_data)
nrow(training_data)

# Pearson Correlation Coefficient
training_EDA <- training_data[c(4,7:26)]
cor(training_EDA, method = "pearson")

# Observe variable distributions data
v_dist <- melt(training_EDA)
ggplot(v_dist,aes(x = value)) + 
  facet_wrap(~variable,scales = "free_x") + 
  geom_histogram()

# Let's create some inital ts to understand the dependent variable total_cases
total_cases_ts <- ts(training_data$total_cases, frequency = 52, start=c(1990,1),
                        end=c(2010,52))
plot.ts(total_cases_ts, ylab='year', xlab='total cases', main='Total Cases of Dengue Fever')
total_components <- decompose(total_cases_ts)
plot(total_components) 

# DATA TRANSFORMATIONS
# Create copy of data for transformations
training_xfer <- training_data
# Apply Box Cox Transformations to variables of interest that don't appear to have a normal distro

# reanalysis_min_air_temp_k
xfer_reanalysis_min_air_temp_k = training_xfer$reanalysis_min_air_temp_k + 1
myt_min=powerTransform(xfer_reanalysis_min_air_temp_k ~ 1)
myt_min$lambda
testTransform(myt_min,myt_min$lambda)
training_xfer$reanalysis_min_air_temp_k_BC = training_xfer$reanalysis_min_air_temp_k^myt_min$lambda
cor(training_xfer$total_cases,training_xfer$reanalysis_min_air_temp_k)
cor(training_xfer$total_cases,training_xfer$reanalysis_min_air_temp_k_BC)

# reanalysis_tdtr_k
xfer_reanalysis_tdtr_k = training_xfer$reanalysis_tdtr_k + 1
myt=powerTransform(xfer_reanalysis_tdtr_k ~ 1)
myt$lambda
testTransform(myt,myt$lambda)
training_xfer$reanalysis_tdtr_k_BC = training_xfer$reanalysis_tdtr_k^myt$lambda
cor(training_xfer$total_cases,training_xfer$reanalysis_tdtr_k)
cor(training_xfer$total_cases,training_xfer$reanalysis_tdtr_k_BC)

# station_min_temp_c
xfer_station_min_temp_c = training_xfer$station_min_temp_c + 1
myt=powerTransform(xfer_station_min_temp_c ~ 1)
myt$lambda
testTransform(myt,myt$lambda)
training_xfer$station_min_temp_c_BC = training_xfer$station_min_temp_c^myt$lambda
cor(training_xfer$total_cases,training_data$station_min_temp_c)
cor(training_xfer$total_cases,training_xfer$station_min_temp_c_BC)

# Create Log version of base dataframe for Neural Network Model
dengue_base_log <- data.frame(total_cases=training_data$total_cases,
                         log.reanalysis_air_temp_k=log(training_data$reanalysis_air_temp_k+1),
                         log.reanalysis_avg_temp_k=log(training_data$reanalysis_avg_temp_k+1),
                         log.reanalysis_min_air_temp_k=log(training_data$reanalysis_min_air_temp_k+1),
                         log.reanalysis_tdtr_k=log(training_data$reanalysis_tdtr_k+1),
                         log.station_diur_temp_rng_c=log(training_data$station_diur_temp_rng_c+1),
                         log.station_min_temp_c=log(training_data$station_min_temp_c+1))

dengue_xfer_log <- data.frame(total_cases=training_xfer$total_cases,
                              log.reanalysis_air_temp_k=log(training_xfer$reanalysis_air_temp_k+1),
                              log.reanalysis_avg_temp_k=log(training_xfer$reanalysis_avg_temp_k+1),
                              log.reanalysis_min_air_temp_k=log(training_xfer$reanalysis_min_air_temp_k+1),
                              log.reanalysis_tdtr_k=log(training_xfer$reanalysis_tdtr_k+1),
                              log.station_diur_temp_rng_c=log(training_xfer$station_diur_temp_rng_c+1),
                              log.station_min_temp_c=log(training_xfer$station_min_temp_c+1))

# MODEL BUILDING
# Baseline Models
# Subset the data into baseline subtraining and subtest
train_subtrain <- head(training_data, 1589)
train_subtest  <- tail(training_data, nrow(training_data) - 1589)

#Split the NN base log dataframe into subtraining and subtest
NN_train_subtrain <- head(dengue_base_log, 1589)
NN_train_subtest  <- tail(dengue_base_log, nrow(dengue_base_log) - 1589)

#Split the NN xfer log dataframe into subtraining and subtest
NN_xfer_train_subtrain <- head(dengue_xfer_log, 1589)
NN_xfer_train_subtest  <- tail(dengue_xfer_log, nrow(dengue_xfer_log) - 1589)


# Baseline model off the highest R-values (>0.1)
fit_1 <- lm(total_cases ~ reanalysis_air_temp_k + reanalysis_avg_temp_k +
                  reanalysis_min_air_temp_k + reanalysis_tdtr_k +
                  station_diur_temp_rng_c + station_min_temp_c, data=train_subtrain)
# Negative Binomial Model
fit_2 <- glm.nb(total_cases ~ reanalysis_air_temp_k + reanalysis_avg_temp_k +
              reanalysis_min_air_temp_k + reanalysis_tdtr_k +
              station_diur_temp_rng_c + station_min_temp_c, data=train_subtrain)
# Poisson Model
fit_3 <- glm(total_cases ~ reanalysis_air_temp_k + reanalysis_avg_temp_k +
                  reanalysis_min_air_temp_k + reanalysis_tdtr_k +
                  station_diur_temp_rng_c + station_min_temp_c, data=train_subtrain,
                  family = 'poisson')
# Neural Network Model w/log transfer
fit_4 <- avNNet(total_cases ~ log.reanalysis_air_temp_k + log.reanalysis_avg_temp_k +
                  log.reanalysis_min_air_temp_k + log.reanalysis_tdtr_k +
                  log.station_diur_temp_rng_c + log.station_min_temp_c, data=NN_train_subtrain,
                repeats=50, size=10, decay=0.1,linout=TRUE)

# Data Transformation Models
# Subset the data into baseline subtraining and subtest
xfer_train_subtrain <- head(training_xfer, 1589)
xfer_train_subtest  <- tail(training_xfer, nrow(training_xfer) - 1589)

# Xfer model off the highest R-values (>0.1)
fit_5 <- lm(total_cases ~ reanalysis_air_temp_k + reanalysis_avg_temp_k +
              reanalysis_min_air_temp_k_BC + reanalysis_tdtr_k_BC +
              station_diur_temp_rng_c + station_min_temp_c_BC, data=xfer_train_subtrain)
# Xfer Negative Binomial Model
fit_6 <- glm.nb(total_cases ~ reanalysis_air_temp_k + reanalysis_avg_temp_k +
                  reanalysis_min_air_temp_k_BC + reanalysis_tdtr_k_BC +
                  station_diur_temp_rng_c + station_min_temp_c_BC, data=xfer_train_subtrain)
# Xfer Poisson Model
fit_7 <- glm(total_cases ~ reanalysis_air_temp_k + reanalysis_avg_temp_k +
               reanalysis_min_air_temp_k_BC + reanalysis_tdtr_k_BC +
               station_diur_temp_rng_c + station_min_temp_c_BC, data=xfer_train_subtrain,
             family = 'poisson')
# Log and Box Cox Xfer Neural Network Model
fit_8 <- avNNet(total_cases ~ log.reanalysis_air_temp_k + log.reanalysis_avg_temp_k +
                  log.reanalysis_min_air_temp_k + log.reanalysis_tdtr_k +
                  log.station_diur_temp_rng_c + log.station_min_temp_c, data=NN_xfer_train_subtrain,
                repeats=50, size=10, decay=0.1,linout=TRUE)

# Box Cox Xfer Neural Network Model (MAE)
fit_9 <- avNNet(total_cases ~ reanalysis_air_temp_k + reanalysis_avg_temp_k +
                  reanalysis_min_air_temp_k + reanalysis_tdtr_k +
                  station_diur_temp_rng_c + station_min_temp_c, data=xfer_train_subtrain,
                repeats=50, size=10, decay=0.1,linout=TRUE)

# Negative Binomial Log & Box Cox transfer
fit_12 <- glm.nb(total_cases ~ log.reanalysis_air_temp_k + log.reanalysis_avg_temp_k +
                   log.reanalysis_min_air_temp_k + log.reanalysis_tdtr_k +
                   log.station_diur_temp_rng_c + log.station_min_temp_c, data=NN_xfer_train_subtrain)

# Arima Model
arima_subtrain <- train_subtrain
arima_total_cases_ts <- ts(arima_subtrain$total_cases, frequency = 52, start=c(1990,1),
                           end=c(2005,52))
plot.ts(arima_total_cases_ts)

# Decompose into seasonal, trend, and noise
arima_components <- decompose(arima_total_cases_ts)
plot(arima_components)

# Seasonally Adjust the TS
arima_total_cases_ts_seasonaladj <- arima_total_cases_ts - arima_components$seasonal
plot.ts(arima_total_cases_ts_seasonaladj)

# Fit the Arima Model 
fit_10 <- auto.arima(arima_total_cases_ts_seasonaladj, seasonal=TRUE)
summary(fit_10) # ARIMA(2,1,3)(0,0,1)
tsdisplay(residuals(fit_10), main="(4,1,4) Model Residuals")
f_cast_1 <- forecast(fit_10, h=52)
plot(f_cast_1)

# Dynamic Regression Model
xreg <- c(training_data$total_cases,
          training_data$reanalysis_air_temp_k,
          training_data$reanalysis_avg_temp_k,
          training_data$reanalysis_min_air_temp_k,
          training_data$reanalysis_tdtr_k,
          training_data$station_diur_temp_rng_c,
          training_data$station_min_temp_c)

fit_11 <- Arima(training_data$total_cases, xreg=training_data$reanalysis_air_temp_k, order=c(2,1,3))
tsdisplay(arima.errors(fit_11), main="Arima Errors")
Box.test(residuals(fit_10), fitdf = 7, lag=10, type="Ljung")
summary(fit_11)

# Hybrid
fit_13 <- glm.nb(total_cases ~ reanalysis_air_temp_k + reanalysis_avg_temp_k +
                reanalysis_min_air_temp_k_BC + reanalysis_tdtr_k +
                station_diur_temp_rng_c + station_min_temp_c, data=xfer_train_subtrain)

# MODEL SELECTION (Model Selection Metric is MAE)
mae <- function(error) return(mean(abs(error)) )
model_score <- function(model, test){
  results <-  predict(model, test)
  score   <-  mae(test$total_cases - results)
  return(score)
}
# Scoring Metric is Mean Absolute Error (MAE)
# Baseline Models
model_score(fit_1,train_subtest) # Linear Regression
model_score(fit_2,train_subtest) # Negative Binomial  SELECTED MODEL
model_score(fit_3,train_subtest) # Poisson
model_score(fit_4,NN_train_subtest) # Neural Network Log Xfer
# Data Transformation Models
model_score(fit_5,xfer_train_subtest) # Linear Regression
model_score(fit_6,xfer_train_subtest) # Negative Binomial
model_score(fit_7,xfer_train_subtest) # Poisson
model_score(fit_8,NN_train_subtest) # Neural Network Log and BoxCox Xfer
model_score(fit_9,xfer_train_subtest) # Neural Network
model_score(fit_13, xfer_train_subtest) # Hybrid
# ARIMA Model
fcast_arima_10 <- forecast(fit_10, h= train_subtest$reanalysis_air_temp_k)
mae_fit10 <- mae(train_subtest$total_cases - fcast_arima_10$x)
mae_fit10

fcast_arima_11 <- forecast(fit_11, xreg= train_subtest$reanalysis_air_temp_k)
mae_fit11 <- mae(train_subtest$total_cases - fcast_arima_11$x)
mae_fit11

# Data Submission Routine
testdata=read.csv(file.choose(), header=TRUE, stringsAsFactors = F)
convert_reanalysis_min_air_temp_k = testdata$reanalysis_min_air_temp_k + 1
myt_TD=powerTransform(convert_reanalysis_min_air_temp_k ~ 1)
testdata$reanalysis_min_air_temp_k_BC <- testdata$reanalysis_min_air_temp_k^myt_TD$lambda
results <- round(predict(fit_13,testdata), digits=0)
scored <- data.frame(city=testdata$city, year=testdata$year, weekofyear=testdata$weekofyear, total_cases=results)
scored$total_cases%<>% na.locf(fromLast = TRUE)
write.csv(scored, "/Users/darenpurnell/Documents/Northwestern MSPA/Predict 413  Applied Time Series and Forecasting/Final_DengAI/submission")

# Selected model is fit_13 with MAE of 12.402 
summary(fit_13)



