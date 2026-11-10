"""
Módulo de Análise de Clusterização
Funções para processamento e análise de dados de futebol
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def create_features(df):
    """
    Cria métricas compostas específicas para futebol
    """
    df_features = df.copy()
    
    # Índice Atlético Geral (força + velocidade normalizada)
    df_features['Indice_Atletico'] = (df_features['Força'] + (6 - df_features['Velocidade'])) / 2
    
    # Índice Técnico (precisão + reflexo)
    df_features['Indice_Tecnico'] = (df_features['Precisão'] + df_features['Reflexo']) / 2
    
    # Índice de Potencial Físico (altura + força normalizada por idade)
    df_features['Potencial_Fisico'] = (df_features['Altura'] + df_features['Força']) / df_features['Idade']
    
    # Agilidade (combinação velocidade + reflexo)
    df_features['Agilidade'] = ((6 - df_features['Velocidade']) + df_features['Reflexo']) / 2
    
    return df_features


def prepare_data_for_clustering(df_features):
    """
    Prepara e normaliza os dados para clustering
    """
    features_for_clustering = [
        'Idade', 'Altura', 'Velocidade', 'Força', 'Reflexo', 'Precisão',
        'Indice_Atletico', 'Indice_Tecnico', 'Potencial_Fisico', 'Agilidade'
    ]
    
    X = df_features[features_for_clustering].copy()
    
    # Normalizando os dados
    scaler = PowerTransformer()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    return X_scaled, scaler, features_for_clustering


def calculate_optimal_clusters(X_scaled, max_k=10):
    """
    Calcula métricas para determinar número ótimo de clusters
    """
    k_range = range(2, max_k + 1)
    
    # K-Means
    kmeans_silhouette = []
    kmeans_davies_bouldin = []
    kmeans_calinski = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        kmeans_silhouette.append(silhouette_score(X_scaled, labels))
        kmeans_davies_bouldin.append(davies_bouldin_score(X_scaled, labels))
        kmeans_calinski.append(calinski_harabasz_score(X_scaled, labels))
    
    # Hierarchical
    hierarchical_silhouette = []
    for k in k_range:
        hierarchical = AgglomerativeClustering(n_clusters=k)
        labels = hierarchical.fit_predict(X_scaled)
        hierarchical_silhouette.append(silhouette_score(X_scaled, labels))
    
    # Gaussian Mixture
    gmm_bic = []
    gmm_aic = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X_scaled)
        gmm_bic.append(gmm.bic(X_scaled))
        gmm_aic.append(gmm.aic(X_scaled))
    
    metrics = {
        'k_range': list(k_range),
        'kmeans': {
            'silhouette': kmeans_silhouette,
            'davies_bouldin': kmeans_davies_bouldin,
            'calinski': kmeans_calinski,
            'best_k': k_range[np.argmax(kmeans_silhouette)]
        },
        'hierarchical': {
            'silhouette': hierarchical_silhouette,
            'best_k': k_range[np.argmax(hierarchical_silhouette)]
        },
        'gmm': {
            'bic': gmm_bic,
            'aic': gmm_aic,
            'best_k': k_range[np.argmin(gmm_bic)]
        }
    }
    
    return metrics


def apply_clustering(X_scaled, k_kmeans, k_hierarchical, k_gmm):
    """
    Aplica os três algoritmos de clustering
    """
    results = {}
    
    # K-Means
    kmeans = KMeans(n_clusters=k_kmeans, random_state=42, n_init=10)
    results['kmeans_labels'] = kmeans.fit_predict(X_scaled)
    results['kmeans_centers'] = kmeans.cluster_centers_
    results['kmeans_model'] = kmeans
    
    # Hierarchical
    hierarchical = AgglomerativeClustering(n_clusters=k_hierarchical)
    results['hierarchical_labels'] = hierarchical.fit_predict(X_scaled)
    
    # Gaussian Mixture
    gmm = GaussianMixture(n_components=k_gmm, covariance_type='full', random_state=42)
    results['gmm_labels'] = gmm.fit_predict(X_scaled)
    results['gmm_model'] = gmm
    
    return results


def interpret_clusters(df_features, n_clusters):
    """
    Interpreta e nomeia os clusters baseado nas características
    """
    perfis = []
    
    for cluster_id in range(n_clusters):
        cluster_data = df_features[df_features['Cluster_KMeans'] == cluster_id]
        
        perfil = {
            'cluster': cluster_id,
            'tamanho': len(cluster_data),
            'idade_media': cluster_data['Idade'].mean(),
            'altura_media': cluster_data['Altura'].mean(),
            'velocidade_media': cluster_data['Velocidade'].mean(),
            'forca_media': cluster_data['Força'].mean(),
            'reflexo_medio': cluster_data['Reflexo'].mean(),
            'precisao_media': cluster_data['Precisão'].mean(),
            'indice_atletico_medio': cluster_data['Indice_Atletico'].mean(),
            'indice_tecnico_medio': cluster_data['Indice_Tecnico'].mean()
        }
        
        perfis.append(perfil)
    
    perfis_df = pd.DataFrame(perfis)
    
    # Interpretar clusters
    interpretacoes = {}
    
    for i, perfil in perfis_df.iterrows():
        cluster_id = int(perfil['cluster'])
        
        # Lógica para classificar baseada nas características
        if perfil['indice_atletico_medio'] > perfis_df['indice_atletico_medio'].mean():
            if perfil['indice_tecnico_medio'] > perfis_df['indice_tecnico_medio'].mean():
                nome = "Completos"
                descricao = "Alunos com bom equilíbrio entre aspectos técnicos e atléticos"
                posicoes = "Meio-campo, Lateral"
                treinamento = "Treinamento balanceado, foco em situações de jogo"
            else:
                nome = "Atletas"
                descricao = "Alunos com destaque em características físicas"
                posicoes = "Zagueiro, Volante"
                treinamento = "Aprimorar aspectos técnicos, manter condicionamento"
        else:
            if perfil['indice_tecnico_medio'] > perfis_df['indice_tecnico_medio'].mean():
                nome = "Técnicos"
                descricao = "Alunos com destaque em habilidades técnicas"
                posicoes = "Meia-atacante, Ponta"
                treinamento = "Desenvolver condicionamento físico, manter técnica"
            else:
                nome = "Em Desenvolvimento"
                descricao = "Alunos em fase de desenvolvimento geral"
                posicoes = "Posições variadas conforme evolução"
                treinamento = "Desenvolvimento geral, fundamentos básicos"
        
        interpretacoes[cluster_id] = {
            'nome': nome,
            'descricao': descricao,
            'posicoes_sugeridas': posicoes,
            'foco_treinamento': treinamento,
            'tamanho': int(perfil['tamanho'])
        }
    
    return perfis_df, interpretacoes


def create_cluster_summary(df_features, features_for_clustering):
    """
    Cria resumo estatístico dos clusters
    """
    cluster_summary = df_features.groupby('Cluster_KMeans')[features_for_clustering].mean()
    return cluster_summary


def get_cluster_examples(df_features, cluster_id, n_examples=5):
    """
    Retorna exemplos de alunos de um cluster específico
    """
    cluster_data = df_features[df_features['Cluster_KMeans'] == cluster_id]
    exemplos = cluster_data.sample(min(n_examples, len(cluster_data)))[
        ['Nome', 'Idade', 'Altura', 'Velocidade', 'Força', 'Reflexo', 'Precisão']
    ]
    return exemplos
