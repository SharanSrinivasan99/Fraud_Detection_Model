#Loading Necessary Packages
library(tidyverse)
library(tidymodels)
library(themis)
library(workflows)
library(glmnet)



#Loading the data
bank_data <- read_csv("E:/Lectures/Q3/704/Project/R code/data/bank_A_transactions_20240731.csv")
bank_data

#-----------------------------------------------------------------------------------------------

# Re-coded variables into binary factors
bank_data$joint_flag <- factor(bank_data$joint_flag)
bank_data$FraudLabel <- factor(bank_data$FraudLabel)

table(bank_data$FraudLabel)
str(bank_data)

new_bank_data <- bank_data

new_bank_data <- within(new_bank_data, {
  CatBalance <- NA # initialize
  CatBalance[balance < 0] <- "Negative"
  CatBalance[balance >= 0] <- "Positive"
}
)
new_bank_data$CatBalance <- 
  factor(new_bank_data$CatBalance, levels = c("Negative", "Positive"))


# -------------------------------------------------------------------------
# Adding the categorical Age
new_bank_data$CatAge <- cut(new_bank_data$Age,
                            breaks = c(0,20,30,40,50,60,70,Inf),
                            labels = c('Under 20', 
                                       '20-30',
                                       '30-40',
                                       '40-50',
                                       '50-60',
                                       '60-70',
                                       '70 & above'))

# -------------------------------------------------------------------------
# Adding the aggregated categorical agent
new_bank_data <- new_bank_data |>
  mutate(agent_cat = 
           case_when(
             grepl("Android", agent, ignore.case = TRUE) ~ "Android",
             grepl("Mozilla", agent, ignore.case = TRUE) ~ "Mozilla",
             grepl("ATM", agent, ignore.case = TRUE) ~ "ATM",
             grepl("iPad", agent, ignore.case = TRUE) ~ "iPad",
             grepl("iPhone", agent, ignore.case = TRUE) ~ "iPhone",
             TRUE ~ agent
           )) |>
  mutate(agent_cat = ifelse(is.na(agent_cat), "Unknown", agent_cat)) 
# replacing N/A with unknown

str(new_bank_data)

# -------------------------------------------------------------------------
# Extracting time from transaction date

new_bank_data$TransactionDate <- 
  as.POSIXct(new_bank_data$TransactionDate) 
# Convert Transaction to timestamp (year, month, date, hour, min, sec)

new_bank_data$TransactionTime <-
  as.numeric(format(new_bank_data$TransactionDate, "%H"))
# filterd timestamp to hours and minutes only


# -------------------------------------------------------------------------
# counting the no. of transactions in 1 hour
# creating a new df: data_time
result <- new_bank_data |>
  group_by(CustomerID, TransactionTime) |>
  summarise(transaction_count = n())

data_time <- merge(new_bank_data, result, by = c("CustomerID", "TransactionTime"), all.x = TRUE)

print(data_time)

# -------------------------------------------------------------------------
# Dropping Some Variables
risk_bank_data <- data_time |> 
  select(-TransactionID, -MerchantLocation,-MerchantID,-agent) 

str(risk_bank_data)
# -------------------------------------------------------------------------
# create a risk variable
#Criteria adjusted after the model
risk_bank_data <- risk_bank_data |> 
  mutate(RiskScore = 
           # High Risk Criteria
           ifelse(Age > 70, 3, 0) +
           ifelse(balance < -10000, 3, 0) +
           ifelse(balance > 90000, 3, 0) +
           ifelse(TransactionAmount > 600, 3, 0) +
           ifelse(FraudLabel == 1, 5, 0) +  
           ifelse(joint_flag == 1, 2, 0) +
           ifelse(agent_cat == "Unknown", 2, 0) +
           ifelse(TransactionType == "transfer", 3, 0) +
           ifelse(transaction_count > 4, 2, 0) +
           ifelse(TransactionTime >= 2 & TransactionTime <= 6, 1, 0) +
           
           # Medium Risk Criteria
           ifelse(Age >= 50 & Age <= 70, 1, 0) +
           ifelse(balance >= -10000 & balance <= -1, 1, 0) +
           ifelse(balance >= 9123 & balance <= 80000, 1, 0) +
           ifelse(TransactionAmount >= 257 & TransactionAmount <= 600, 1, 0) +
           ifelse(FraudLabel == 0, 1, 0) +
           ifelse(joint_flag == 0, 1, 0) +
           ifelse(agent_cat %in% c("Ipad", "Iphone", "Android", "Mozilla"), 1, 0) +
           ifelse(TransactionType == "Purchase", 1, 0) +
           ifelse(transaction_count >= 3 & transaction_count <= 4, 1, 0) +
           ifelse(TransactionTime >= 6 & TransactionTime <= 10, 1, 0)
  )

# Define thresholds for risk categories
risk_bank_data <- risk_bank_data |> 
  mutate(RiskCategory = case_when(
    RiskScore >= 18 ~ "High Risk",  # You can adjust this threshold
    RiskScore >= 15 & RiskScore <18 ~ "Medium Risk",
    TRUE ~ "Low Risk"
  ))
# counting the frequency of risk category
table(risk_bank_data$RiskCategory)
View(risk_bank_data)

#Changing risk category to factor
risk_bank_data$RiskCategory<-as.factor(risk_bank_data$RiskCategory)
new_data<-risk_bank_data %>% 
  mutate(across(where(is.character),as.factor)) %>%
  mutate(across(where(is.integer),as.factor)) %>% 
  mutate(across(where(is.POSIXct),as.factor)) %>% 
  mutate(across(where(is.factor),as.numeric))

#Selecting Variables
new_data<-new_data %>%
  select(Age,balance,TransactionAmount,FraudLabel,joint_flag,TransactionTime)

str(new_data)

new_data <- new_data %>%
  mutate(FraudLabel = as.factor(FraudLabel))

table(new_data$FraudLabel)

new_data <- new_data %>%
  mutate(FraudLabel = case_when(
    FraudLabel == "2" ~ 1,      
    FraudLabel == "1" ~ 0,   
    TRUE ~ NA_real_                  
  )) %>%
  mutate(FraudLabel = factor(FraudLabel, levels = c(0, 1)))

table(new_data$FraudLabel)

str(new_data)

#---------------------------------------------------------------------------------------
#LogisticRegressionModel


#Splitting
set.seed(123)
data_split <- initial_split(new_data, prop = 0.75, strata = FraudLabel)
train_data <- training(data_split)
test_data <- testing(data_split)

# Create a recipe with upsampling
fraud_recipe <- recipe(FraudLabel ~ ., data = train_data) %>%
  step_upsample(FraudLabel, over_ratio = 1) %>%  # Upsampling
  step_normalize(all_numeric_predictors()) %>%   
  step_dummy(all_nominal_predictors())           

# Specify logistic regression model with regularization
log_reg_model <- logistic_reg(penalty = 0.01, mixture = 1) %>%  # L1 regularization (Lasso)
  set_engine("glmnet") 

# Create a workflow
fraud_workflow <- workflow() %>%
  add_model(log_reg_model) %>%
  add_recipe(fraud_recipe)

# Training the model using cross-validation
set.seed(123)
fraud_fit <- fraud_workflow %>%
  fit_resamples(
    resamples = vfold_cv(train_data, v = 5),  
    metrics = metric_set(roc_auc, accuracy, sensitivity, specificity)
  )

# Fitting the final model on the entire training set
final_fit <- fraud_workflow %>%
  last_fit(data_split)

# Extracting the trained workflow
final_workflow <- extract_workflow(final_fit)

fitted_model <- extract_fit_parsnip(final_fit)
coef_df <- tidy(fitted_model$fit)
print(coef_df)


# Important Predictors
coef_df <- data.frame(
  term = c("(Intercept)", "Age", "balance", "TransactionAmount", "joint_flag", "TransactionTime"),
  estimate = c(-0.0292, 1.09, -0.0232, -0.144, 0.0120, -0.0960)
)

#absolute values of coefficients
coef_df$abs_estimate <- abs(coef_df$estimate)

#total sum of absolute coefficients
total_abs_estimate <- sum(coef_df$abs_estimate)

#Percentage contribution of each variable
coef_df$percentage_contribution <- (coef_df$abs_estimate / total_abs_estimate) * 100

#results
print(coef_df)


# Predicted probabilities 
pred_probs <- predict(final_workflow, new_data = test_data, type = "prob")
predicted_classes <- ifelse(pred_probs$.pred_1 > 0.5, "1", "0")
predicted_classes <- factor(predicted_classes, levels = c("0", "1"))
log_final <- data.frame(FraudLabel = test_data$FraudLabel, predicted = predicted_classes)

# Confusion Matrix
conf_mat <- conf_mat(log_final, truth = FraudLabel, estimate = predicted)

# Calculate metrics again
metrics <- metrics(log_final, truth = FraudLabel, estimate = predicted)

# Print the confusion matrix and metrics
print(conf_mat)
print(metrics)

#----------------------------------------------------------
#Visualization
#Boxplot of Fraud and Age 
box_plot_age <- new_data 

box_plot_age$NewFraudLabel <- ifelse(box_plot_age$FraudLabel == 1, "Fraud", "No Fraud")

box_plot_age |>
  ggplot(aes(y = Age, x = NewFraudLabel, fill = NewFraudLabel)) + geom_boxplot() +
  theme(plot.title = element_text(hjust=0.5),
        panel.background = element_rect(fill='transparent'), 
        plot.background = element_rect(fill='transparent', color=NA), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(), 
        legend.background = element_rect(fill='transparent'),
        legend.box.background = element_rect(fill='transparent'),
        size = 1,
        panel.border = element_rect(color = "black", 
                                    fill = NA, 
                                    size = 1),
        legend.position = "none",
        axis.text.x = element_text(face = "bold"),  
        axis.text.y = element_text(face = "bold"),
        plot.text.x = element_text(face = "bold"),  
        plot.text.y = element_text(face = "bold")
  ) +
  labs(x = "Fraud Label", y = "Age") 

View(risk_bank_data)


#Visualization For Transaction Hour
fraud_data <- risk_bank_data %>% filter(FraudLabel == 1)

hourly_transactions_fraud <- fraud_data %>%
  group_by(TransactionTime) %>%
  summarise(transaction_count = n())

# Visualization: Number of transactions by hour with a centered title (for fraud cases)
ggplot(hourly_transactions_fraud, aes(x = TransactionTime, y = transaction_count)) +
  geom_line() +
  geom_point() +
  ggtitle("Fraudulent Transactions by Hour of the Day") +
  xlab("Hour of the Day") +
  ylab("Number of Fraudulent Transactions") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_x_continuous(breaks = 0:23) + 
  theme(plot.title = element_text(hjust=0.5),
        panel.background = element_rect(fill='transparent'), 
        plot.background = element_rect(fill='transparent', color=NA), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(), 
        legend.background = element_rect(fill='transparent'),
        legend.box.background = element_rect(fill='transparent'),
        size = 1,
        panel.border = element_rect(color = "black", 
                                    fill = NA, 
                                    size = 1),
        legend.position = "none"
  )


#-------------------------------------------------------------------------------
#Unused XGB Model
# Define the recipe for feature engineering
fraud_recipe <- recipe(FraudLabel ~ ., data = train_data) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())

gbm_model <- boost_tree(
  trees = 500,         # Reduce the number of trees
  min_n = 5,
  tree_depth = 6,
  learn_rate = 0.01
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

fraud_workflow <- workflow() %>%
  add_model(gbm_model) %>%
  add_recipe(fraud_recipe)

set.seed(123)
gbm_fit <- fraud_workflow %>%
  fit(data = train_data)

pred_probs <- predict(gbm_fit, new_data = test_data, type = "prob")


predicted_classes <- ifelse(pred_probs$.pred_1 > 0.5, "1", "0")

# Convert to factor
predicted_classes <- factor(predicted_classes, levels = c("0", "1"))

# Create a data frame with actual and predicted values
results <- data.frame(FraudLabel = test_data$FraudLabel, predicted = predicted_classes)

# Confusion Matrix
conf_mat <- conf_mat(results, truth = FraudLabel, estimate = predicted)

# Calculate metrics
metrics <- metrics(results, truth = FraudLabel, estimate = predicted)

# Print confusion matrix and metrics
print(conf_mat)
print(metrics)



