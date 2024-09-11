# Function to install and load a package if not already installed
install_if_required <- function(package) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, dependencies = TRUE)
    library(package, character.only = TRUE)
  }
}

# Install and Load necessary libraries
install_if_required("tidyverse")
install_if_required("fastDummies")
install_if_required("caret")
install_if_required("pROC")

# Define the file path
file_path <- "porto-seguro-safe-driver-prediction/train.csv"

# Read the CSV file
data <- readr::read_csv(file_path)

# Replace -1 with NA
data <- data %>%
  dplyr::mutate(across(everything(), ~ dplyr::na_if(., -1)))

# Remove rows with NAs
data <- tidyr::drop_na(data)

# Identify and convert columns ending in "cat" to factors
cols_to_factor <- grep("target|cat$", names(data), value = TRUE)
data <- data %>%
  dplyr::mutate(across(all_of(cols_to_factor), as.factor))

# Convert remaining columns to numeric
cols_to_numeric <- setdiff(names(data), cols_to_factor)
data <- data %>%
  dplyr::mutate(across(all_of(cols_to_numeric), as.numeric))

# Create dummy variables for categorical columns
data <- fastDummies::dummy_cols(data, select_columns = cols_to_factor)

# Filter only numeric columns
numeric_data <- data %>%
  dplyr::select(where(is.numeric))

# Display the structure of the final data
utils::str(numeric_data)

# Normalize the numeric columns using scale
normalized_data <- numeric_data %>%
  dplyr::mutate(across(everything(), ~ scale(.) %>% as.vector()))

# PCA using prcomp
pca_result <- stats::prcomp(normalized_data, center = TRUE, scale. = TRUE)

# Plot the scree plot
graphics::plot(pca_result, type = "l")

# Select the number of principal components to use (e.g., the first 10 components)
num_components <- 10
pca_data <- pca_result$x[, 1:num_components]

# Add the target column to the principal components
data_pca <- base::data.frame(target = data$target, pca_data)

# Ensure trainControl is set to save class probabilities
train_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, savePredictions = TRUE)

# Convert class levels to valid R variable names
levels(data_pca$target) <- make.names(levels(data_pca$target))

# Define a seed for reproducibility
base::set.seed(5997760) 

# Train the model using k-fold cross-validation
model_pca_cv <- train(target ~ ., data = data_pca, method = "glm", family = binomial, trControl = train_control, metric = "ROC")

# Print the cross-validation results
base::print(model_pca_cv)

# Plot the ROC curve
pROC::plot.roc(model_pca_cv$pred$obs, model_pca_cv$pred$glm)