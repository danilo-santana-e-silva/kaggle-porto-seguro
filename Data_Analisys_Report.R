# Função para instalar e carregar um pacote, se ainda não estiver instalado
install_if_required <- function(package) {
    if (!require(package, character.only = TRUE)) {
        install.packages(package, dependencies = TRUE)
        library(package, character.only = TRUE)
    }
}

# Instalar e carregar as bibliotecas necessárias
install_if_required("tidyverse")
install_if_required("caret")
install_if_required("pROC")

# Definir o caminho do arquivo
file_path <- "porto-seguro-safe-driver-prediction/train.csv"

# Ler o arquivo CSV
data <- readr::read_csv(file_path)

# Substituir -1 por NA
data <- dplyr::mutate(data, across(everything(), ~ dplyr::na_if(., -1)))

# Remover linhas com NAs (considere a imputação como alternativa)
data <- tidyr::drop_na(data)

# Identificar e converter colunas terminadas em "cat" e "bin" para fatores
cols_to_factor <- grep("bin$|cat$", names(data), value = TRUE)
data <- dplyr::mutate(data, across(all_of(cols_to_factor), as.factor))

# Converter as colunas restantes para numéricas
cols_to_numeric <- setdiff(names(data), cols_to_factor)
data <- dplyr::mutate(data, across(all_of(cols_to_numeric), as.numeric))

# Reservar target
target <- data$target

# Criar dummy variables para as colunas categóricas
dummy_model <- caret::dummyVars(target ~ ., data = data, fullRank = TRUE)
data <- predict(dummy_model, newdata = data)
data <- as.data.frame(data)
data$target <- target

# Filtrar apenas as colunas numéricas
numeric_data <- dplyr::select(data, where(is.numeric))

# PCA usando prcomp
pca_result <- stats::prcomp(numeric_data, center = TRUE, scale. = TRUE)

# Plotar o gráfico de scree
graphics::plot(pca_result, type = "l", main = "Scree Plot")

# Selecionar o número de componentes principais a ser utilizado (por exemplo, 10 componentes)
num_components <- 35
pca_data <- pca_result$x[, 1:num_components]

# Adicionar a coluna target aos componentes principais
data_pca <- base::data.frame(target = data$target, pca_data)

# Definir uma semente para reprodutibilidade
base::set.seed(5997760)

train_index <- createDataPartition(data_pca$target, p = 0.8, list = FALSE)
train_data <- data_pca[train_index, ]
test_data <- data_pca[-train_index, ]

# Renomear os níveis da variável target para nomes válidos em R
train_data$target <- factor(make.names(train_data$target))
test_data$target <- factor(make.names(test_data$target))

# Definir controle para a busca de hiperparâmetros
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Testar diferentes modelos e ajustar hiperparâmetros
models <- list(
  glm = train(target ~ ., data = train_data, method = "glm", family = binomial, metric = "ROC", trControl = ctrl)
  #rf = train(target ~ ., data = train_data, method = "rf", metric = "ROC", trControl = ctrl)
  #xgb = train(target ~ ., data = train_data, method = "xgbTree", metric = "ROC", trControl = ctrl)
)

# Escolher o melhor modelo com base no AUC
best_model <- models[[which.max(sapply(models, function(m) max(m$results$ROC)))]]

# Fazer previsões com o melhor modelo
predictions <- predict(best_model, newdata = test_data, type = "prob")

# Calcular o AUC
roc_curve <- roc(test_data$target, predictions[, 2])
auc_value <- auc(roc_curve)

# Imprimir o valor do AUC
print(paste("AUC:", auc_value))

# Plotar a curva ROC
plot.roc(roc_curve, main = paste("ROC Curve (AUC =", round(auc_value, 2), ")"))
