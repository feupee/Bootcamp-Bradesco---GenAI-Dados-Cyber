import pandas as pd

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

#Acurácia: porcentagem de detecções corretas
#Recall: quantas fraudes o modelo conseguiu identificar
#Precision: o que o modelo chamou de fraude era realmente fraude

df = pd.read_csv(url) #Carrega o dataset de trans

# print(df.head())

#Problema de classificação desbalanceada, pois temos muito mais transações legítimas do que fraudulentas
print(df["Class"].value_counts(normalize=True))

#feature engineering: É mais importante que o modelo, o processo de criar ou transformar as variáveis para melhores o desempenho do modelo


import numpy as np
df["Amount_log"] = np.log1p(df["Amount"]) #log1p evitar problemas com valores zero, pois log(0) é indefinido


#ajustando os dados para uma escala comum, o que ajuda a melhorar o desempenho de muitos algoritmos de machine learning
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount_log"]]) #Escala os valores para que tenham média 0 e desvio padrão 1

from sklearn.model_selection import train_test_split
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42) #stratify=y garante que a proporção de classes seja mantida no conjunto de treino e teste

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000) #max_iter define o número máximo de iterações para o algoritmo de otimização
model.fit(X_train, y_train) #Treina o modelo com os dados de treino
y_pred = model.predict(X_test) #Faz previsões com os dados de teste

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred)) #Exibe métricas de avaliação do modelo, como precisão, recall e f1-score

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_probs = model.predict_proba(X_test)[:, 1] #Obtém as probabilidades de previsão para a classe positiva (fraude)
fpr, tpr, _ = roc_curve(y_test, y_probs) #Calcula a curva ROC
plt.plot(fpr, tpr)
plt.title("Curva ROC")
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.show()

print("AUC: ", roc_auc_score(y_test, y_probs)) #Calcula a área sob a curva ROC (AUC), que é uma métrica de desempenho do modelo

from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_probs) #Calcula a curva de precisão-recall
plt.plot(recall, precision)
plt.title("Curva Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

#Undersampling
fraudes = df[df["Class"] == 1] #Seleciona apenas as transações fraudulentas
normais = df[df["Class"] == 0] #Seleciona apenas as transações normais

df_under = pd.concat([fraudes,normais])

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_res, y_res = smote.fit_resample(X, y)

#Ramdom Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight="balanced", n_jobs=-1, random_state=42) #class_weight="balanced" ajusta os pesos das classes para lidar com o desbalanceamento
rf.fit(X_res, y_res) #Treina o modelo com os dados balanceados
y_pred_rf = rf.predict(X_test) #Faz previsões com os dados de teste

print(classification_report(y_test, y_pred_rf)) #Exibe métricas de avaliação do modelo Random Forest

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train) #Treina o pipeline com os dados de treino
y_pred_pipeline = pipeline.predict(X_test) #Faz previsões com os dados de teste

threshold = 0.3
y_pred_custom = (y_probs > threshold).astype(int) #Aplica um limiar personalizado para classificar as previsões

print(classification_report(y_test, y_pred_custom)) #Exibe métricas de avaliação do modelo com o limiar personalizado

from xgboost import XGBClassifier

xgb = XGBClassifier(scale_pos_weight=10, use_label_encoder=False, eval_metric="logloss") #scale_pos_weight ajusta o peso da classe positiva para lidar com o desbalanceamento
xgb.fit(X_res, y_res) #Treina o modelo XGBoost com os dados balanceados
y_pred_xgb = xgb.predict(X_test) #Faz previsões com os dados de teste

#Importância das variáveis

import matplotlib.pyplot as plt
importancias = xgb.feature_importances_ #Obtém a importância das variáveis do modelo XGBoost
plt.bar(range(len(importancias)), importancias) #Cria um gráfico de barras com a importância das variáveis
plt.title("Importância das Variáveis")
plt.show()

# Ajuste de hiperparâmetros
from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100],
}

grid = GridSearchCV(
    estimator=XGBClassifier(
        eval_metric="logloss"
    ),
    param_grid=param_grid,
    scoring="recall",
    cv=3
)

grid.fit(X_train, y_train)

print("Melhores parâmetros:", grid.best_params_)

#Explicabilidade
import shap
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test[:100])

shap.plots.bar(shap_values)