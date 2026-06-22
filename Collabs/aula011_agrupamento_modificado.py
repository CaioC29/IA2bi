# -*- coding: utf-8 -*-
"""Aula011-Agrupamento.ipynb - VERSÃO MODIFICADA

Versão com dataset expandido e parâmetros ajustados
"""

# !pip install -q scikit-learn pandas matplotlib

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# DATASET EXPANDIDO - De 20 para 30 pessoas
dados = pd.DataFrame({
    "Pessoa": [f"P{i}" for i in range(1, 31)],

    # idade em anos
    "Idade": [18, 20, 22, 25, 27, 30, 35, 38, 40, 42,
              50, 52, 55, 58, 60, 65, 68, 70, 72, 75,
              23, 28, 33, 45, 62, 44, 36, 51, 67, 26],

    # horas de sono por noite
    "Sono": [8, 7, 8, 7, 6, 7, 6, 5, 6, 5,
             6, 5, 5, 4, 5, 4, 4, 5, 4, 3,
             8, 7, 6, 5, 4, 6, 5, 5, 4, 7],

    # exposição a ambientes fechados por dia
    "AmbienteFechado": [1, 2, 1, 2, 3, 2, 5, 6, 5, 7,
                        6, 7, 8, 8, 7, 9, 9, 8, 10, 9,
                        2, 3, 4, 7, 9, 6, 5, 7, 8, 2],

    # frequência de atividade física semanal
    "AtividadeFisica": [5, 4, 5, 4, 3, 4, 2, 1, 2, 1,
                        2, 1, 1, 0, 1, 0, 0, 1, 0, 0,
                        5, 4, 3, 2, 0, 2, 3, 1, 0, 4],

    # número de sintomas recentes
    "Sintomas": [0, 1, 0, 1, 1, 0, 2, 3, 2, 3,
                 3, 4, 4, 5, 4, 5, 6, 5, 6, 7,
                 0, 1, 2, 3, 5, 2, 1, 4, 6, 1],

    # tomou vacina? 1 = sim, 0 = não
    "Vacinado": [1, 1, 1, 1, 0, 1, 1, 0, 1, 0,
                 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                 1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
})

print("=== DATASET CARREGADO ===")
print(f"Total de pessoas: {len(dados)}")
print(dados.head(10))

atributos = [
    "Idade",
    "Sono",
    "AmbienteFechado",
    "AtividadeFisica",
    "Sintomas",
    "Vacinado"
]

X = dados[atributos]

print("\n=== DADOS ORIGINAIS (primeiras 5 linhas) ===")
print(X.head())

# Normalização
normalizador = StandardScaler()
X_normalizado = normalizador.fit_transform(X)

print("\n=== DADOS NORMALIZADOS (primeiras 5 linhas) ===")
print(X_normalizado[:5])

# K-MEANS - Parâmetros modificados
# n_clusters: 3 → 4 (mais grupos)
# n_init: 10 → 20 (mais tentativas para melhor convergência)
print("\n" + "="*80)
print("K-MEANS CLUSTERING")
print("="*80)

kmeans = KMeans(
    n_clusters=4,          # modificado de 3
    random_state=42,
    n_init=20              # modificado de 10
)

dados["Cluster_KMeans"] = kmeans.fit_predict(X_normalizado)

print("\nAgrupamentos KMeans:")
print(dados[["Pessoa", "Cluster_KMeans"]])
print(f"\nInércia: {kmeans.inertia_:.2f}")

# Testando múltiplos valores de k para encontrar o melhor
print("\n=== TESTE DE MÚLTIPLOS K (Elbow Method) ===")
inercias = []
silhuetas = []
k_values = range(2, 8)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_normalizado)
    inercias.append(km.inertia_)
    silhuetas.append(silhouette_score(X_normalizado, labels))
    print(f"k={k}: Inércia={km.inertia_:.2f}, Silhueta={silhouette_score(X_normalizado, labels):.3f}")

# AGNES - Parâmetros modificados
# linkage: "ward" → "complete" (método de ligação diferente)
print("\n" + "="*80)
print("AGNES (AGGLOMERATIVE CLUSTERING)")
print("="*80)

agnes = AgglomerativeClustering(
    n_clusters=4,          # modificado para corresponder ao K-Means
    linkage="complete"     # modificado de "ward"
)

dados["Cluster_AGNES"] = agnes.fit_predict(X_normalizado)

print("\nAgrupamentos AGNES:")
print(dados[["Pessoa", "Cluster_AGNES"]])

# DBSCAN - Parâmetros modificados
# eps: 1.6 → 1.8 (raio maior, menos pontos isolados)
# min_samples: 2 → 3 (critério mais rigoroso)
print("\n" + "="*80)
print("DBSCAN (DENSITY-BASED CLUSTERING)")
print("="*80)

dbscan = DBSCAN(
    eps=1.8,               # modificado de 1.6
    min_samples=3          # modificado de 2
)

dados["Cluster_DBSCAN"] = dbscan.fit_predict(X_normalizado)

print("\nAgrupamentos DBSCAN:")
print(dados[["Pessoa", "Cluster_DBSCAN"]])

n_clusters_dbscan = len(set(dados["Cluster_DBSCAN"])) - (1 if -1 in dados["Cluster_DBSCAN"] else 0)
n_noise = list(dados["Cluster_DBSCAN"]).count(-1)
print(f"Número de clusters: {n_clusters_dbscan}")
print(f"Pontos de ruído: {n_noise}")

# Tabela consolidada
print("\n" + "="*80)
print("CONSOLIDAÇÃO DE TODOS OS ALGORITMOS")
print("="*80)
print(dados[[
    "Pessoa",
    "Idade",
    "Sono",
    "AmbienteFechado",
    "AtividadeFisica",
    "Sintomas",
    "Vacinado",
    "Cluster_KMeans",
    "Cluster_AGNES",
    "Cluster_DBSCAN"
]])

# Cálculo de Silhueta
def calcular_silhouette(nome_algoritmo, labels):
    grupos = set(labels)

    if len(grupos) <= 1:
        return None

    grupos_reais = [g for g in grupos if g != -1]

    if len(grupos_reais) < 2:
        return None

    return silhouette_score(X_normalizado, labels)


resultados = pd.DataFrame({
    "Algoritmo": ["K-Means", "AGNES", "DBSCAN"],
    "Silhouette": [
        calcular_silhouette("K-Means", dados["Cluster_KMeans"]),
        calcular_silhouette("AGNES", dados["Cluster_AGNES"]),
        calcular_silhouette("DBSCAN", dados["Cluster_DBSCAN"])
    ]
})

print("\n" + "="*80)
print("MÉTRICAS DE QUALIDADE DOS CLUSTERS")
print("="*80)
print(resultados)

# PCA para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalizado)

dados["PCA1"] = X_pca[:, 0]
dados["PCA2"] = X_pca[:, 1]

print(f"\nVariância explicada pelo PCA: {pca.explained_variance_ratio_.sum():.2%}")
print(f"PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.2%}")

# Visualizações
print("\n" + "="*80)
print("GERANDO GRÁFICOS...")
print("="*80)

# Gráfico 1: K-Means
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    dados["PCA1"],
    dados["PCA2"],
    c=dados["Cluster_KMeans"],
    cmap="viridis",
    s=100,
    alpha=0.6,
    edgecolors='black'
)

for i, pessoa in enumerate(dados["Pessoa"]):
    plt.text(dados["PCA1"][i], dados["PCA2"][i], pessoa, fontsize=8, ha='center')

plt.colorbar(scatter, label="Cluster")
plt.title("Agrupamento com K-Means (k=4)")
plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/kmeans_cluster.png', dpi=100, bbox_inches='tight')
plt.show()

# Gráfico 2: AGNES
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    dados["PCA1"],
    dados["PCA2"],
    c=dados["Cluster_AGNES"],
    cmap="plasma",
    s=100,
    alpha=0.6,
    edgecolors='black'
)

for i, pessoa in enumerate(dados["Pessoa"]):
    plt.text(dados["PCA1"][i], dados["PCA2"][i], pessoa, fontsize=8, ha='center')

plt.colorbar(scatter, label="Cluster")
plt.title("Agrupamento com AGNES (linkage=complete)")
plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/agnes_cluster.png', dpi=100, bbox_inches='tight')
plt.show()

# Gráfico 3: DBSCAN
plt.figure(figsize=(8, 6))
colors = dados["Cluster_DBSCAN"]
scatter = plt.scatter(
    dados["PCA1"],
    dados["PCA2"],
    c=colors,
    cmap="coolwarm",
    s=100,
    alpha=0.6,
    edgecolors='black'
)

for i, pessoa in enumerate(dados["Pessoa"]):
    plt.text(dados["PCA1"][i], dados["PCA2"][i], pessoa, fontsize=8, ha='center')

plt.colorbar(scatter, label="Cluster (-1 = Ruído)")
plt.title("Agrupamento com DBSCAN (eps=1.8, min_samples=3)")
plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/dbscan_cluster.png', dpi=100, bbox_inches='tight')
plt.show()

# Gráfico 4: Elbow Method
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(k_values, inercias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.title('Elbow Method - Inércia')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(k_values, silhuetas, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Coeficiente de Silhueta')
plt.title('Silhueta para diferentes valores de k')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/elbow_silhueta.png', dpi=100, bbox_inches='tight')
plt.show()

# Análise de perfis - K-Means
print("\n" + "="*80)
print("ANÁLISE DE PERFIS - K-MEANS")
print("="*80)

perfil_kmeans = dados.groupby("Cluster_KMeans")[atributos].mean()
print(perfil_kmeans)

for cluster, linha in perfil_kmeans.iterrows():
    print(f"\n{'='*60}")
    print(f"Cluster {cluster} - K-Means")
    print(f"Idade média: {linha['Idade']:.1f} anos")
    print(f"Sono médio: {linha['Sono']:.1f} horas/noite")
    print(f"Ambiente fechado: {linha['AmbienteFechado']:.1f} horas/dia")
    print(f"Atividade física: {linha['AtividadeFisica']:.1f} vezes/semana")
    print(f"Sintomas médios: {linha['Sintomas']:.1f}")
    print(f"Taxa de vacinação: {linha['Vacinado']*100:.0f}%")

# Análise de perfis - AGNES
print("\n" + "="*80)
print("ANÁLISE DE PERFIS - AGNES")
print("="*80)

perfil_agnes = dados.groupby("Cluster_AGNES")[atributos].mean()
print(perfil_agnes)

for cluster, linha in perfil_agnes.iterrows():
    print(f"\n{'='*60}")
    print(f"Cluster {cluster} - AGNES")
    print(f"Idade média: {linha['Idade']:.1f} anos")
    print(f"Sono médio: {linha['Sono']:.1f} horas/noite")
    print(f"Ambiente fechado: {linha['AmbienteFechado']:.1f} horas/dia")
    print(f"Atividade física: {linha['AtividadeFisica']:.1f} vezes/semana")
    print(f"Sintomas médios: {linha['Sintomas']:.1f}")
    print(f"Taxa de vacinação: {linha['Vacinado']*100:.0f}%")

# Análise de perfis - DBSCAN
print("\n" + "="*80)
print("ANÁLISE DE PERFIS - DBSCAN")
print("="*80)

perfil_dbscan = dados.groupby("Cluster_DBSCAN")[atributos].mean()
print(perfil_dbscan)

for cluster, linha in perfil_dbscan.iterrows():
    if cluster == -1:
        nome_cluster = "Ruído / Casos Atípicos"
    else:
        nome_cluster = f"Cluster {cluster}"

    print(f"\n{'='*60}")
    print(f"{nome_cluster} - DBSCAN")
    print(f"Idade média: {linha['Idade']:.1f} anos")
    print(f"Sono médio: {linha['Sono']:.1f} horas/noite")
    print(f"Ambiente fechado: {linha['AmbienteFechado']:.1f} horas/dia")
    print(f"Atividade física: {linha['AtividadeFisica']:.1f} vezes/semana")
    print(f"Sintomas médios: {linha['Sintomas']:.1f}")
    print(f"Taxa de vacinação: {linha['Vacinado']*100:.0f}%")

# Função de interpretação melhorada
def interpretar_clusters(dados, coluna_cluster, nome_algoritmo):
    perfis = dados.groupby(coluna_cluster)[atributos].mean()

    print(f"\n{'='*80}")
    print(f"INTERPRETAÇÃO DETALHADA - {nome_algoritmo}")
    print(f"{'='*80}")

    for cluster, linha in perfis.iterrows():
        print(f"\n{'-'*60}")

        if cluster == -1:
            print("Grupo -1: CASOS ATÍPICOS (detectados pelo DBSCAN)")
        else:
            print(f"Grupo {cluster}")

        interpretacao = []

        if linha["Idade"] >= 55:
            interpretacao.append("pessoas MAIS VELHAS (55+)")
        elif linha["Idade"] <= 30:
            interpretacao.append("pessoas MAIS JOVENS (≤30)")
        else:
            interpretacao.append("pessoas de idade INTERMEDIÁRIA")

        if linha["Sono"] < 5:
            interpretacao.append("POUCARAS HORAS de sono (<5h)")
        elif linha["Sono"] >= 7:
            interpretacao.append("SONO ADEQUADO (≥7h)")
        else:
            interpretacao.append("sono moderado (5-7h)")

        if linha["AmbienteFechado"] >= 7:
            interpretacao.append("ALTA exposição a ambientes fechados")
        elif linha["AmbienteFechado"] <= 3:
            interpretacao.append("BAIXA exposição a ambientes fechados")
        else:
            interpretacao.append("exposição moderada a ambientes fechados")

        if linha["AtividadeFisica"] <= 1:
            interpretacao.append("BAIXA atividade física")
        elif linha["AtividadeFisica"] >= 4:
            interpretacao.append("ALTA atividade física")
        else:
            interpretacao.append("atividade física moderada")

        if linha["Sintomas"] >= 5:
            interpretacao.append("MUITOS sintomas recentes")
        elif linha["Sintomas"] <= 1:
            interpretacao.append("POUCOS sintomas recentes")
        else:
            interpretacao.append("sintomas moderados")

        if linha["Vacinado"] >= 0.7:
            interpretacao.append("MAIORIA VACINADA")
        elif linha["Vacinado"] <= 0.3:
            interpretacao.append("MAIORIA NÃO VACINADA")
        else:
            interpretacao.append("vacinação mista")

        print("Perfil: " + ", ".join(interpretacao))

interpretar_clusters(dados, "Cluster_KMeans", "K-MEANS")
interpretar_clusters(dados, "Cluster_AGNES", "AGNES")
interpretar_clusters(dados, "Cluster_DBSCAN", "DBSCAN")

print("\n" + "="*80)
print("FIM DA ANÁLISE")
print("="*80)
