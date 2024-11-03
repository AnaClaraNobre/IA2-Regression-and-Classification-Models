import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados
file_path = 'C:/Users/anacl/OneDrive/Documentos/Pessoal/UNIFOR/IA2_regressaoEClassificacao/aerogerador.dat'
data = pd.read_csv(file_path, sep='\t', header=None)
data.columns = ['Velocidade_Vento', 'Potencia_Gerada']

# Visualização inicial dos dados
plt.scatter(data['Velocidade_Vento'], data['Potencia_Gerada'])
plt.xlabel('Velocidade do Vento')
plt.ylabel('Potência Gerada')
plt.title('Relação entre Velocidade do Vento e Potência Gerada')
plt.show()

X = data[['Velocidade_Vento']].values  
y = data['Potencia_Gerada'].values 

## 4.1 MQO (Mínimos Quadrados Ordinários)
def mqo(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return beta[0], beta[1]  

intercept_mqo, coef_mqo = mqo(X, y)
print(f"Intercepto (MQO): {intercept_mqo}")
print(f"Coeficiente (MQO): {coef_mqo}")

## 4.2 MQO Regularizado (Tikhonov ou Ridge Regression)
def ridge_regression(X, y, lambda_):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    identity = np.eye(X_b.shape[1])
    identity[0, 0] = 0  # Não regularizar o termo de intercepto
    beta = np.linalg.inv(X_b.T.dot(X_b) + lambda_ * identity).dot(X_b.T).dot(y)
    return beta[0], beta[1]  

# Lista de valores de lambda conforme especificado no enunciado, incluindo lambda = 1
lambdas = [0.25, 0.5, 0.75, 1]
ridge_results = []

for l in lambdas:
    intercept_ridge, coef_ridge = ridge_regression(X, y, l)
    ridge_results.append((l, intercept_ridge, coef_ridge))
    print(f"Lambda: {l}, Intercepto: {intercept_ridge}, Coeficiente: {coef_ridge}")


## 4.3 Média dos Valores Observáveis
def mean_model(y):
    return np.mean(y)

mean_value = mean_model(y)
print(f"Média dos Valores Observáveis: {mean_value}")

# 5. Validação com Simulação de Monte Carlo
R = 500  
rss_results = {l: [] for l in ['MQO', 'Média'] + lambdas}

for _ in range(R):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_size = int(len(X) * 0.8)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # MQO
    intercept_mqo, coef_mqo = mqo(X_train, y_train)
    predictions_mqo = intercept_mqo + coef_mqo * X_test.flatten()
    rss_results['MQO'].append(np.sum((y_test - predictions_mqo) ** 2))
    
    # Média dos valores observáveis
    mean_value = mean_model(y_train)
    predictions_mean = np.full_like(y_test, mean_value)
    rss_results['Média'].append(np.sum((y_test - predictions_mean) ** 2))
    
    # MQO Regularizado
    for l in lambdas:
        intercept_ridge, coef_ridge = ridge_regression(X_train, y_train, l)
        predictions_ridge = intercept_ridge + coef_ridge * X_test.flatten()
        rss_results[l].append(np.sum((y_test - predictions_ridge) ** 2))


# 6. Análise dos Resultados
model_names = {
    'Média': 'Média da variável dependente',
    'MQO': 'MQO tradicional',
    0.25: 'MQO regularizado (0,25)',
    0.5: 'MQO regularizado (0,5)',
    0.75: 'MQO regularizado (0,75)',
    1: 'MQO regularizado (1)'
}

metrics = []
for model, rss_list in rss_results.items():
    metrics.append({
        'Modelos': model_names.get(model, model),
        'Média': np.mean(rss_list),
        'Desvio-Padrão': np.std(rss_list),
        'Maior Valor': np.max(rss_list),
        'Menor Valor': np.min(rss_list)
    })

metrics_df = pd.DataFrame(metrics)
metrics_df = metrics_df[['Modelos', 'Média', 'Desvio-Padrão', 'Maior Valor', 'Menor Valor']]


fig, ax = plt.subplots(figsize=(10, 4)) 
ax.axis('tight')
ax.axis('off')

table = ax.table(
    cellText=metrics_df.values,
    colLabels=metrics_df.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(metrics_df.columns))))  

plt.title("Resultados da Validação com Monte Carlo - Tabela de Métricas")
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
metrics_df.plot(
    x='Modelos', 
    y=['Média', 'Desvio-Padrão', 'Maior Valor', 'Menor Valor'], 
    kind='bar', 
    ax=ax
)
plt.title("Análise de Desempenho dos Modelos de Regressão")
plt.xlabel("Modelos")
plt.ylabel("Valores de RSS")
plt.legend(loc="upper right", title="Métricas")
plt.xticks(rotation=45, ha="right")  
plt.tight_layout()  
plt.show()
