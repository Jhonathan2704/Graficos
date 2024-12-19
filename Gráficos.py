import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Lendo o arquivo
df = pd.read_csv('ecommerce_estatistica.csv')

# Configurando o estilo dos gráficos
plt.style.use('seaborn')

# 1. Histograma da distribuição de Preços
plt.figure(figsize=(10, 6))
plt.hist(df['Preço'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribuição dos Preços dos Produtos')
plt.xlabel('Preço (R$)')
plt.ylabel('Frequência')
plt.show()

# 2. Gráfico de Dispersão entre Preço e Número de Avaliações
plt.figure(figsize=(10, 6))
plt.scatter(df['N_Avaliações'], df['Preço'], alpha=0.5)
plt.title('Relação entre Preço e Número de Avaliações')
plt.xlabel('Número de Avaliações')
plt.ylabel('Preço (R$)')
plt.show()

# 3. Mapa de Calor das correlações
correlation_matrix = df[['Preço', 'Nota', 'N_Avaliações', 'Desconto']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Mapa de Calor - Correlações')
plt.show()

# 4. Gráfico de Barras - Média de preço por Gênero
plt.figure(figsize=(10, 6))
df.groupby('Gênero')['Preço'].mean().plot(kind='bar')
plt.title('Preço Médio por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Preço Médio (R$)')
plt.show()

# 5. Gráfico de Pizza - Distribuição de Produtos por Temporada
plt.figure(figsize=(10, 8))
df['Temporada'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribuição de Produtos por Temporada')
plt.ylabel('')
plt.show()

# 6. Gráfico de Densidade - Distribuição das Notas
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df['Nota'], fill=True)
plt.title('Distribuição de Densidade das Notas')
plt.xlabel('Nota')
plt.ylabel('Densidade')
plt.show()

# 7. Gráfico de Regressão - Relação entre Preço e Desconto
plt.figure(figsize=(10, 6))
sns.regplot(x='Desconto', y='Preço', data=df)
plt.title('Regressão: Relação entre Preço e Desconto')
plt.xlabel('Desconto (%)')
plt.ylabel('Preço (R$)')
plt.show()

print("Análise concluída com sucesso!")