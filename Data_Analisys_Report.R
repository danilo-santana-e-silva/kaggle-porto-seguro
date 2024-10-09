# Fazer previsões com o melhor modelo
predictions <- predict(best_model, newdata = test_data, type = "prob")

# Calcular o AUC
roc_curve <- roc(test_data$target, predictions[, 2])
auc_value <- auc(roc_curve)

# Imprimir o valor do AUC
print(paste("AUC:", auc_value))

# Plotar a curva ROC
plot.roc(roc_curve, main = paste("ROC Curve (AUC =", round(auc_value, 2), ")"))

# Criar predictions_factors com base em predictions[, 2]
predictions_factors <- dplyr::if_else(predictions[, 2] < 0.5, "X0", "X1")

#as factor
predictions_factors <- as.factor(predictions_factors)
caret::confusionMatrix(predictions_factors,test_data$target)

# Plotar a matriz de confusão
conf_matrix <- caret::confusionMatrix(predictions_factors, test_data$target)
conf_matrix_table <- as.table(conf_matrix)
fourfoldplot(conf_matrix_table, color = c("#CC6666", "#99CC99"), conf.level = 0, margin = 1, main = "Confusion Matrix")
