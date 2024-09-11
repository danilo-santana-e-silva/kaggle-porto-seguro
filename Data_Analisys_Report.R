# Function to install and load a package if not already installed
install_if_required <- function(package) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, dependencies = TRUE)
    library(package, character.only = TRUE)
  }
}

# Install and Load necessary libraries
install_if_required("tidyverse")
install_if_required("caret")
install_if_required("pROC")

# Define the file path
file_path <- "porto-seguro-safe-driver-prediction/train.csv"

# Read the CSV file
data <- readr::read_csv(file_path)

# Replace -1 with NA
data <- dplyr::mutate(data, across(everything(), ~ dplyr::na_if(., -1)))

# Remove rows with NAs (consider imputing instead)
data <- tidyr::drop_na(data)

# Identify and convert columns ending in "cat" and "bin" to factors
cols_to_factor <- grep("target|bin$|cat$", names(data), value = TRUE)
data <- dplyr::mutate(data, across(all_of(cols_to_factor), as.factor))

# Convert remaining columns to numeric
cols_to_numeric <- setdiff(names(data), cols_to_factor)
data <- dplyr::mutate(data, across(all_of(cols_to_numeric), as.numeric))

# Create dummy variables and add the target column back
dummy_data <- caret::dummyVars(~ . - target, data = data) %>%
  predict(newdata = data) %>%
  as.data.frame() %>%
  cbind(target = data$target)

# Replace the original data with the dummy data
data <- dummy_data

# Filter only numeric columns
numeric_data <- dplyr::select(data, where(is.numeric))

# Display the structure of the final data
utils::str(numeric_data)

# Normalize the numeric columns using caret
pre_proc_values <- caret::preProcess(numeric_data, method = c("center", "scale"))
normalized_data <- stats::predict(pre_proc_values, numeric_data)

# PCA using prcomp
pca_result <- stats::prcomp(normalized_data, center = TRUE, scale. = TRUE)

# Plot the scree plot
graphics::plot(pca_result, type = "l")

# Select the number of principal components to use (e.g., 10 components)
num_components <- 10
pca_data <- pca_result$x[, 1:num_components]

# Add the target column to the principal components
data_pca <- base::data.frame(target = data$target, pca_data)

# Convert class levels to valid R variable names
levels(data_pca$target) <- base::make.names(levels(data_pca$target))

# Define a seed for reproducibility
base::set.seed(5997760)

# Split the data into training and testing sets (80:20 ratio)
train_index <- caret::createDataPartition(data_pca$target, p = 0.8, list = FALSE)
train_data <- data_pca[train_index, ]
test_data <- data_pca[-train_index, ]

# Train the model on the training set
model_pca <- caret::train(target ~ .,
                          data = train_data,
                          method = "glm",
                          family = binomial)

# Print the training results
base::print(model_pca)

# Make predictions on the testing set
predictions <- stats::predict(model_pca, newdata = test_data, type = "prob")

# Ensure the predictor is numeric
predictions <- base::as.numeric(predictions[,2])

# Calculate AUC value
auc_value <- pROC::roc(test_data$target, predictions)$auc

# Plot the ROC curve
pROC::plot.roc(test_data$target, predictions, 
               main = base::paste("ROC Curve (AUC =", base::round(auc_value, 2),")"))

# Print the AUC value
base::print(auc_value)