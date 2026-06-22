# -*- coding: utf-8 -*-
"""Aula010-AprioriDidatico.ipynb - VERSÃO MODIFICADA

Versão com mais dados e parâmetros ajustados
"""

# !pip install -q mlxtend

import pandas as pd
import numpy as np

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

import logging
import warnings
import os

# reduz logs gerais
logging.getLogger().setLevel(logging.CRITICAL)

# reduz logs de bibliotecas comuns
for nome in ["matplotlib", "PIL", "numexpr", "tensorflow", "absl"]:
    logging.getLogger(nome).setLevel(logging.CRITICAL)

# ignora warnings
warnings.filterwarnings("ignore")

# TensorFlow, se estiver no notebook
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# DATASET EXPANDIDO - Aumentei de 10 para 20 alunos
dados = pd.DataFrame({
    "Aluno": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
              "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20"],
    "Frequencia": ["Alta", "Alta", "Média", "Baixa", "Baixa", "Alta", "Média", "Baixa", "Alta", "Média",
                   "Alta", "Alta", "Baixa", "Média", "Média", "Alta", "Baixa", "Média", "Alta", "Alta"],
    "HorasEstudo": ["Muitas", "Moderadas", "Moderadas", "Poucas", "Poucas", "Muitas", "Poucas", "Moderadas", "Muitas", "Moderadas",
                    "Muitas", "Moderadas", "Poucas", "Moderadas", "Moderadas", "Muitas", "Poucas", "Poucas", "Muitas", "Moderadas"],
    "Internet": ["Boa", "Boa", "Regular", "Ruim", "Ruim", "Boa", "Regular", "Regular", "Boa", "Boa",
                 "Boa", "Boa", "Regular", "Boa", "Regular", "Boa", "Ruim", "Regular", "Boa", "Boa"],
    "Resultado": ["Aprovado", "Aprovado", "Aprovado", "Reprovado", "Reprovado", "Aprovado", "Reprovado", "Reprovado", "Aprovado", "Aprovado",
                  "Aprovado", "Aprovado", "Reprovado", "Aprovado", "Aprovado", "Aprovado", "Reprovado", "Reprovado", "Aprovado", "Aprovado"]
})

print("=== DADOS DOS ALUNOS ===")
print(dados)
print(f"\nTotal de alunos: {len(dados)}")
print(f"\nDistribuição de Aprovados/Reprovados:")
print(dados["Resultado"].value_counts())

# Transformando em transações
transacoes = []

for _, linha in dados.iterrows():
    transacao = [
        f"Frequencia={linha['Frequencia']}",
        f"HorasEstudo={linha['HorasEstudo']}",
        f"Internet={linha['Internet']}",
        f"Resultado={linha['Resultado']}"
    ]
    transacoes.append(transacao)

print("\n=== EXEMPLO DE TRANSAÇÕES ===")
for i, transacao in enumerate(transacoes[:3]):
    print(f"Aluno A{i+1}: {transacao}")

# Transformador de transações
te = TransactionEncoder()
matriz = te.fit(transacoes).transform(transacoes)
dados_transformados = pd.DataFrame(matriz, columns=te.columns_)

print("\n=== MATRIZ TRANSFORMADA ===")
print(dados_transformados.head())

# APRIORI - Parâmetros modificados
# min_support: 0.3 → 0.25 (mais items frequentes)
itemsets_frequentes = apriori(
    dados_transformados,
    min_support=0.25,      # modificado de 0.3
    use_colnames=True
)

print("\n=== ITEMSETS FREQUENTES ===")
print(f"Total de itemsets frequentes: {len(itemsets_frequentes)}")
itemsets_ord = itemsets_frequentes.sort_values(by="support", ascending=False)
print(itemsets_ord)

# REGRAS DE ASSOCIAÇÃO - Parâmetros modificados
# metric: 'confidence' (mantém)
# min_threshold: 0.7 → 0.6 (mais regras descobertas)
regras = association_rules(
    itemsets_frequentes,
    metric="confidence",
    min_threshold=0.6      # modificado de 0.7
)

print(f"\n=== REGRAS DE ASSOCIAÇÃO DESCOBERTAS ===")
print(f"Total de regras: {len(regras)}")

if len(regras) > 0:
    # Resumo com mais métricas
    regras_resumidas = regras[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].copy()
    
    regras_resumidas = regras_resumidas.sort_values(by="confidence", ascending=False)
    
    print("\nRegras ordenadas por confiança:")
    print(regras_resumidas)
    
    # Função de formatação
    def formatar_conjunto(conjunto):
        return ", ".join(list(conjunto))
    
    # Adicionando coluna "Regra"
    regras_resumidas["Regra"] = regras_resumidas.apply(
        lambda linha: f"{formatar_conjunto(linha['antecedents'])} → {formatar_conjunto(linha['consequents'])}",
        axis=1
    )
    
    print("\n=== REGRAS FORMATADAS ===")
    resultado_final = regras_resumidas[["Regra", "support", "confidence", "lift"]].sort_values(
        by="confidence",
        ascending=False
    )
    print(resultado_final)
    
    # ANÁLISE DE APROVAÇÃO
    print("\n" + "="*80)
    print("REGRAS PARA APROVAÇÃO")
    print("="*80)
    
    regras_aprovacao = regras_resumidas[
        regras_resumidas["consequents"].apply(lambda x: "Resultado=Aprovado" in x)
    ]
    
    if len(regras_aprovacao) > 0:
        print(regras_aprovacao[["Regra", "support", "confidence", "lift"]].sort_values(
            by="confidence",
            ascending=False
        ))
    else:
        print("Nenhuma regra encontrada para aprovação com os parâmetros atuais.")
    
    # ANÁLISE DE REPROVAÇÃO
    print("\n" + "="*80)
    print("REGRAS PARA REPROVAÇÃO")
    print("="*80)
    
    regras_reprovacao = regras_resumidas[
        regras_resumidas["consequents"].apply(lambda x: "Resultado=Reprovado" in x)
    ]
    
    if len(regras_reprovacao) > 0:
        print(regras_reprovacao[["Regra", "support", "confidence", "lift"]].sort_values(
            by="confidence",
            ascending=False
        ))
    else:
        print("Nenhuma regra encontrada para reprovação com os parâmetros atuais.")
    
    # ANÁLISE ADICIONAL - Lift e Leverage
    print("\n" + "="*80)
    print("ANÁLISE ADICIONAL - MÉTRICAS DETALHADAS")
    print("="*80)
    
    regras_detalhes = regras.copy()
    regras_detalhes["Regra"] = regras_detalhes.apply(
        lambda linha: f"{formatar_conjunto(linha['antecedents'])} → {formatar_conjunto(linha['consequents'])}",
        axis=1
    )
    
    # Adicionar mais métricas
    regras_detalhes["leverage"] = regras["lift"] - 1  # Aproximação
    
    print("\nRegras com as melhores pontuações Lift (>1 = melhor que aleatório):")
    top_lift = regras_detalhes[["Regra", "confidence", "lift"]].nlargest(5, "lift")
    print(top_lift)
    
else:
    print("Nenhuma regra de associação foi descoberta com os parâmetros atuais.")
    print("Tente reduzir min_support ou min_threshold.")

print("\n" + "="*80)
print("FIM DA ANÁLISE")
print("="*80)
