library(mlbench)
library(purrr)
library(xgboost)
library(dplyr)

data("PimaIndiansDiabetes2")
ds <- as.data.frame(na.omit(PimaIndiansDiabetes2))
## fit a logistic regression model to obtain a parametric equation
logmodel <- glm(diabetes ~ .,
                data = ds,
                family = "binomial")
summary(logmodel)

cfs <- coefficients(logmodel) ## extract the coefficients
prednames <- variable.names(ds)[-9] ## fetch the names of predictors in a vector
prednames

sz <- 100000000 ## to be used in sampling
##sample(ds$pregnant, size = sz, replace = T)

dfdata <- map_dfc(prednames,
                  function(nm){ ## function to create a sample-with-replacement for each pred.
                    eval(parse(text = paste0("sample(ds$",nm,
                                             ", size = sz, replace = T)")))
                  }) ## map the sample-generator on to the vector of predictors
## and combine them into a dataframe

names(dfdata) <- prednames
dfdata

class(cfs[2:length(cfs)])

length(cfs)
length(prednames)
## Next, compute the logit values
pvec <- map((1:8),
            function(pnum){
              cfs[pnum+1] * eval(parse(text = paste0("dfdata$",
                                                     prednames[pnum])))
            }) %>% ## create beta[i] * x[i]
  reduce(`+`) + ## sum(beta[i] * x[i])
  cfs[1] ## add the intercept

## exponentiate the logit to obtain probability values of thee outcome variable
dfdata$outcome <- ifelse(1/(1 + exp(-(pvec))) > 0.5,
                         1, 0)


set.seed(123)
sizes <- c(100, 1000, 10000, 100000, 1000000, 10000000)
results <- data.frame()
for (sz in sizes) {
  cat("Dataset size:", sz, "\n")
  dsmall <- dfdata %>% slice_sample(n = sz)
  n_train <- floor(0.8 * sz)
  train_idx <- sample(1:sz, n_train)
  
  train_x <- as.matrix(dsmall[train_idx, prednames])
  train_y <- dsmall$outcome[train_idx]
  
  test_x <- as.matrix(dsmall[-train_idx, prednames])
  test_y <- dsmall$outcome[-train_idx]
  
  dtrain <- xgb.DMatrix(data = train_x, label = train_y)
  
  start_time <- Sys.time()
  
  model <- xgboost(data = dtrain,
                   objective = "binary:logistic",
                   nrounds = 50,
                   max_depth = 3,
                   eta = 0.3,
                   verbose = 0)
  
  end_time <- Sys.time()
  time_taken <- as.numeric(difftime(end_time, start_time, units = "secs"))
  preds <- predict(model, test_x)
  preds_class <- ifelse(preds > 0.5, 1, 0)
  accuracy <- mean(preds_class == test_y)
  results <- rbind(results,
                   data.frame(Method = "XGBoost",
                              Dataset_Size = sz,
                              Accuracy = round(accuracy, 4),
                              Time_Taken_Sec = round(time_taken, 2)))
}
print(results)


library(caret)
set.seed(123)

sizes <- c(100, 1000, 10000, 100000, 1000000, 10000000)
results_caret <- data.frame()

for (sz in sizes) {
  cat("Dataset size:", sz, "\n")
  
  # Sample the data
  dsmall <- dfdata %>% slice_sample(n = sz)
  
  # Split into 80% train and 20% test
  n_train <- floor(0.8 * sz)
  train_idx <- sample(1:sz, n_train)
  
  train_data <- dsmall[train_idx, ]
  test_data <- dsmall[-train_idx, ]
  
  # Convert outcome to factor for classification
  train_data$outcome <- as.factor(train_data$outcome)
  test_data$outcome <- as.factor(test_data$outcome)
  
  # Set up training control for 5-fold CV
  trctrl <- trainControl(method = "cv", number = 5)
  
  # Time the model fitting
  start_time <- Sys.time()
  
  model <- train(
    outcome ~ ., 
    data = train_data,
    method = "xgbTree",
    trControl = trctrl,
    tuneLength = 3
  )
  
  end_time <- Sys.time()
  
  time_taken <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  # Predict
  preds <- predict(model, newdata = test_data)
  
  accuracy <- mean(preds == test_data$outcome)
  
  # Save results
  results_caret <- rbind(results_caret,
                         data.frame(Method = "XGBoost (caret)",
                                    Dataset_Size = sz,
                                    Accuracy = round(accuracy, 4),
                                    Time_Taken_Sec = round(time_taken, 2)))
}

print(results_caret)
