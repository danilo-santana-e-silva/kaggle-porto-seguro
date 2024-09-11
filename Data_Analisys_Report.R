install.packages("fastDummies")
library(fastDummies)

# Definir o caminho do arquivo
file_path <- "porto-seguro-safe-driver-prediction/train.csv"
data <- read.csv(file_path)

# Identificar e definir colunas terminadas em "bin" e "cat" como fatores
cols_to_factor <- grep("target|bin$|cat$", names(data), value = TRUE)
data[cols_to_factor] <- lapply(data[cols_to_factor], as.factor)

# Identificar e definir colunas terminadas em "ind" e "calc" como ordinais
columns_to_order <- grep("ps_ind_0[13]|ps_ind_01[45]|ps_car_11$|ps_calc_0[456789]|ps_calc_1[01234]", names(data), value = TRUE)
data[columns_to_order] <- lapply(data[columns_to_order], as.ordered)

# Definir as demais colunas como numéricas
cols_to_numeric <- setdiff(names(data), c(cols_to_factor, columns_to_order))
data[cols_to_numeric] <- lapply(data[cols_to_numeric], as.numeric)

# Substituir -1 por NA
data[data == -1] <- NA

# Remover linhas com NAs
data <- na.omit(data)

# Criar variáveis dummy para colunas categóricas
data <- dummy_cols(data, select_columns = cols_to_factor)

#########################################
# Criar uma partição de treino e teste
set.seed(5997760) # Define a semente para reprodução dos resultados
train_indices <- sample(1:nrow(data), size = 0.7 * nrow(data), replace = FALSE) # 70% dos dados para treino
train_data <- data[train_indices, ] # Dados de treino
test_data <- data[-train_indices, ] # Dados de teste

# Aplicar GLM
model <- glm(target ~ ., data = train_data, family = binomial)

# Imprimir o resumo do modelo
summary(model)

# Predict the target variable on the test data
predictions <- predict(model, newdata = test_data, type = "response")

# Calculate AUC
library(pROC)
roc_curve <- roc(test_data$target, predictions)
auc_value <- auc(roc_curve)

# Print the performance metrics
cat("Accuracy:", accuracy, "\n")
cat("AUC:", auc_value, "\n")

# Plotar a curva ROC
plot(roc_curve, main = paste("ROC Curve (AUC =", round(auc_value, 2), ")"))

str(data)
##########################################
# Selecionar apenas as colunas numéricas para a PCA
numeric_data <- data[cols_to_numeric]
pca_result <- prcomp(numeric_data, center = TRUE, scale. = TRUE)
plot(pca_result, type = "l")

# Selecionar o número de componentes principais a serem usados (por exemplo, os primeiros 10 componentes)
num_components <- 5
pca_data <- pca_result$x[, 1:num_components]

# Adicionar a coluna de destino aos componentes principais
data_pca <- data.frame(target = data$target, pca_data)

# Ajustar um modelo de regressão logística binária utilizando apenas os componentes principais
model_pca <- glm(target ~ ., data = data_pca, family = binomial)

# Imprimir o resumo do modelo
summary(model_pca)

##############################################################################################
# Instalar e carregar pacotes necessários
install.packages("pROC")
library(pROC)

# Transformar os dados de teste usando PCA
test_data[cols_to_factor] <- lapply(test_data[cols_to_factor], as.factor)
test_data[columns_to_order] <- lapply(test_data[columns_to_order], as.ordered)
test_data[cols_to_numeric] <- lapply(test_data[cols_to_numeric], as.numeric)
test_data[test_data == -1] <- NA
test_data <- na.omit(test_data)
test_data <- dummy_cols(test_data, select_columns = cols_to_factor)
numeric_test_data <- test_data[cols_to_numeric]
pca_test_data <- predict(pca_result, newdata = numeric_test_data)
pca_test_data <- pca_test_data[, 1:num_components]

# Adicionar a coluna de destino aos componentes principais dos dados de teste
test_data_pca <- data.frame(target = test_data$target, pca_test_data)

# Prever os valores usando o modelo ajustado
predictions <- predict(model_pca, newdata = test_data_pca, type = "response")

# Calcular a AUC
roc_curve <- roc(test_data_pca$target, predictions)
auc_value <- auc(roc_curve)

# Plotar a curva ROC
plot(roc_curve, main = paste("ROC Curve (AUC =", round(auc_value, 2), ")"))