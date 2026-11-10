# -*- coding: utf-8 -*-
"""
Sistema de Análise de Clusterização para Escola de Futebol Guerra
Desenvolvido com Streamlit para análise de perfis de atletas
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import sys
import os
import warnings
import logging

# Configurar path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))

import clustering_analysis as ca

# Configuração do Plotly para evitar warnings
pio.templates.default = "plotly_white"

# Suprimir warnings do Plotly
warnings.filterwarnings('ignore', message='.*keyword arguments have been deprecated.*')

# Suprimir logs de depreciação do Streamlit/Plotly
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Configuração da página
st.set_page_config(
    page_title="Análise de Clusterização - Escola de Futebol Guerra",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_sample_data():
    """Carrega dados de exemplo"""
    sample_path = os.path.join(current_dir, 'data', 'raw', 'base_criancas_futebol.csv')
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    return None


def validate_dataframe(df):
    """Valida colunas necessárias"""
    required = ['Nome', 'Idade', 'Altura', 'Velocidade', 'Força', 'Reflexo', 'Precisão']
    missing = [col for col in required if col not in df.columns]
    return len(missing) == 0, missing


def initialize_session_state():
    """Inicializa variáveis de sessão"""
    defaults = {
        'df': None,
        'df_features': None,
        'X_scaled': None,
        'scaler': None,
        'features_for_clustering': None,
        'metrics': None,
        'results': None,
        'k_kmeans': None,
        'perfis_df': None,
        'interpretacoes': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def auto_calculate_metrics(df):
    """Calcula métricas automaticamente quando dados são carregados"""
    # Verificar se precisa recalcular (dados novos ou ainda não calculados)
    needs_calculation = (
        st.session_state.metrics is None or 
        st.session_state.df is None or 
        len(st.session_state.df) != len(df)
    )
    
    if needs_calculation:
        with st.spinner("🔄 Calculando métricas automaticamente..."):
            df_features = ca.create_features(df)
            X_scaled, scaler, features = ca.prepare_data_for_clustering(df_features)
            metrics = ca.calculate_optimal_clusters(X_scaled, max_k=10)
            
            st.session_state.df = df.copy()
            st.session_state.df_features = df_features
            st.session_state.X_scaled = X_scaled
            st.session_state.scaler = scaler
            st.session_state.features_for_clustering = features
            st.session_state.metrics = metrics


# Título principal
st.title("Análise de Clusterização - Escola de Futebol Guerra")
st.markdown("**Sistema de Identificação de Perfis e Otimização de Treinamento Esportivo**")
st.markdown("---")

# Inicializar session state
initialize_session_state()

# ====================
# SIDEBAR
# ====================

with st.sidebar:
    st.header("Configurações")
    
    uploaded_file = st.file_uploader(
        "Selecione um arquivo CSV",
        type=['csv'],
        help="Faça upload do arquivo com dados dos atletas"
    )
    
    df = None
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Arquivo carregado: {len(df)} registros")
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {str(e)}")
    else:
        use_sample = st.checkbox("Utilizar dados de exemplo", value=True)
        if use_sample:
            df = load_sample_data()
            if df is not None:
                st.success(f"Dados de exemplo: {len(df)} atletas")
            else:
                st.warning("Dados de exemplo não disponíveis")
        else:
            st.info("Aguardando upload de arquivo CSV")
    
    if df is not None:
        is_valid, missing = validate_dataframe(df)
        if not is_valid:
            st.error(f"Colunas faltando: {', '.join(missing)}")
            df = None
        else:
            # Calcular métricas automaticamente
            auto_calculate_metrics(df)
            
            st.markdown("---")
            st.subheader("Informações")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Atletas", len(df))
            with col2:
                st.metric("Colunas", len(df.columns))
            
            # Indicador de que métricas foram calculadas
            if st.session_state.metrics is not None:
                st.success("✅ Métricas calculadas automaticamente")

# ====================
# CONTEÚDO PRINCIPAL
# ====================

if df is None:
    st.info("Por favor, carregue um dataset para começar a análise")
    st.stop()

# Armazenar no session state
st.session_state.df = df

# Criar tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Exploração",
    "2. Análise",
    "3. Clustering",
    "4. Resultados",
    "5. Download"
])

# ====================
# TAB 1: EXPLORAÇÃO
# ====================

with tab1:
    st.header("Exploração dos Dados")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Dataset")
        st.dataframe(df.head(15), width='stretch', height=400)
    
    with col2:
        st.subheader("Estatísticas")
        st.dataframe(df.describe().round(2), width='stretch')
    
    st.markdown("---")
    st.subheader("Distribuições")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(df, x='Idade', title='Idade', color_discrete_sequence=['#3b82f6'])
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    
    with col2:
        fig = px.histogram(df, x='Altura', title='Altura', color_discrete_sequence=['#10b981'])
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    
    with col3:
        fig = px.histogram(df, x='Velocidade', title='Velocidade', color_discrete_sequence=['#f59e0b'])
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    
    st.markdown("---")
    st.subheader("Correlação")
    
    chars = ['Idade', 'Altura', 'Velocidade', 'Força', 'Reflexo', 'Precisão']
    corr = df[chars].corr()
    
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig.update_layout(height=500)
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    
    st.markdown("---")
    
    # Boxplot comparativo
    st.subheader("Análise Comparativa - Boxplot")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chars = ['Altura', 'Velocidade', 'Força', 'Reflexo', 'Precisão']
        df_melt = df[chars].melt(var_name='Característica', value_name='Valor')
        
        fig = px.box(
            df_melt,
            x='Característica',
            y='Valor',
            color='Característica',
            title='Boxplot: Distribuição das Características'
        )
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    
    with col2:
        st.markdown("""
        **Dica Interpretação do Boxplot:**
        
        - **Caixa**: Representa 50% dos dados (Q1 a Q3)
        - **Linha central**: Mediana (Q2)
        - **Whiskers**: Extensão até 1.5 x IQR
        - **Pontos**: Outliers potenciais
        
        Use este gráfico para identificar:
        - Dispersão dos dados
        - Valores atípicos
        - Simetria da distribuição
        """)

# ====================
# TAB 2: ANÁLISE
# ====================

with tab2:
    st.header("Análise de Clusters")
    
    # As métricas são calculadas automaticamente após o upload
    if st.session_state.metrics is not None:
        metrics = st.session_state.metrics
        
        st.markdown("---")
        st.subheader("Número Ótimo de Clusters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("K-Means", f"{metrics['kmeans']['best_k']} clusters")
        
        with col2:
            st.metric("Hierárquico", f"{metrics['hierarchical']['best_k']} clusters")
        
        with col3:
            st.metric("GMM", f"{metrics['gmm']['best_k']} componentes")
        
        st.markdown("---")
        st.subheader("Gráficos de Métricas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=metrics['k_range'], y=metrics['kmeans']['silhouette'],
                                    mode='lines+markers', name='K-Means'))
            fig.add_trace(go.Scatter(x=metrics['k_range'], y=metrics['hierarchical']['silhouette'],
                                    mode='lines+markers', name='Hierárquico'))
            fig.update_layout(title='Silhouette Score', xaxis_title='k', yaxis_title='Score')
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=metrics['k_range'], y=metrics['gmm']['bic'],
                                    mode='lines+markers', name='BIC'))
            fig.add_trace(go.Scatter(x=metrics['k_range'], y=metrics['gmm']['aic'],
                                    mode='lines+markers', name='AIC'))
            fig.update_layout(title='GMM - BIC/AIC', xaxis_title='k', yaxis_title='Score')
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

        
        st.markdown("---")
        st.subheader("Visualização 3D - PCA")
        
        # Aplicar PCA para visualização 3D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(st.session_state.X_scaled)
        
        # Criar DataFrame temporário para visualização
        df_viz = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'PC3': X_pca[:, 2],
            'Nome': df['Nome'].values
        })
        
        # Criar gráfico 3D interativo
        fig = go.Figure(data=[go.Scatter3d(
            x=df_viz['PC1'],
            y=df_viz['PC2'],
            z=df_viz['PC3'],
            mode='markers',
            marker=dict(
                size=5,
                color=df_viz['PC1'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="PC1")
            ),
            text=df_viz['Nome'],
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'Visualização 3D - Análise de Componentes Principais<br><sub>Variância explicada: {pca.explained_variance_ratio_.sum():.2%}</sub>',
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'
            ),
            height=600
        )
        
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        # Explicação detalhada dos componentes PCA
        st.markdown("### Composição dos Componentes Principais")
        
        # Obter as features originais
        feature_names = st.session_state.X_scaled.columns.tolist()
        
        # Criar DataFrame com os loadings (contribuições)
        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=['PC1', 'PC2', 'PC3'],
            index=feature_names
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**PC1** ({pca.explained_variance_ratio_[0]:.1%} da variância)")
            loadings_pc1 = loadings_df['PC1'].sort_values(ascending=False)
            st.dataframe(loadings_pc1.round(3), width='stretch')
            
        with col2:
            st.markdown(f"**PC2** ({pca.explained_variance_ratio_[1]:.1%} da variância)")
            loadings_pc2 = loadings_df['PC2'].sort_values(ascending=False)
            st.dataframe(loadings_pc2.round(3), width='stretch')
            
        with col3:
            st.markdown(f"**PC3** ({pca.explained_variance_ratio_[2]:.1%} da variância)")
            loadings_pc3 = loadings_df['PC3'].sort_values(ascending=False)
            st.dataframe(loadings_pc3.round(3), width='stretch')
        
        st.info("""
        **Como interpretar os componentes:**
        - Valores **positivos altos** indicam que a feature contribui fortemente na direção positiva do componente
        - Valores **negativos altos** indicam contribuição forte na direção negativa
        - Valores **próximos de zero** indicam pouca relevância para aquele componente
        - PC1 geralmente captura a maior variação nos dados (características que mais diferenciam os atletas)
        """)
    else:
        st.info("📊 Carregue os dados na barra lateral para visualizar as análises automaticamente.")

# ====================
# TAB 3: CLUSTERING
# ====================

with tab3:
    st.header("Aplicar Clustering")
    
    if st.session_state.metrics is None:
        st.warning("Execute a análise na aba 'Análise' primeiro")
    else:
        metrics = st.session_state.metrics
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            k_kmeans = st.number_input("K-Means: k", 2, 10, metrics['kmeans']['best_k'])
        
        with col2:
            k_hier = st.number_input("Hierárquico: k", 2, 10, metrics['hierarchical']['best_k'])
        
        with col3:
            k_gmm = st.number_input("GMM: k", 2, 10, metrics['gmm']['best_k'])
        
        if st.button("Executar Clustering", type="primary", width='stretch'):
            with st.spinner("Aplicando clustering..."):
                results = ca.apply_clustering(st.session_state.X_scaled, k_kmeans, k_hier, k_gmm)
                
                df_features = st.session_state.df_features.copy()
                df_features['Cluster_KMeans'] = results['kmeans_labels']
                df_features['Cluster_Hierarchical'] = results['hierarchical_labels']
                df_features['Cluster_GMM'] = results['gmm_labels']
                
                perfis_df, interpretacoes = ca.interpret_clusters(df_features, k_kmeans)
                
                st.session_state.results = results
                st.session_state.df_features = df_features
                st.session_state.k_kmeans = k_kmeans
                st.session_state.perfis_df = perfis_df
                st.session_state.interpretacoes = interpretacoes
            
            st.success("Clustering aplicado!")
        
        if st.session_state.results is not None:
            st.markdown("---")
            st.subheader("Distribuição dos Clusters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                dist = pd.Series(st.session_state.results['kmeans_labels']).value_counts().sort_index()
                fig = px.bar(x=dist.index, y=dist.values, title='K-Means')
                st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
            with col2:
                dist = pd.Series(st.session_state.results['hierarchical_labels']).value_counts().sort_index()
                fig = px.bar(x=dist.index, y=dist.values, title='Hierárquico')
                st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
            with col3:
                dist = pd.Series(st.session_state.results['gmm_labels']).value_counts().sort_index()
                fig = px.bar(x=dist.index, y=dist.values, title='GMM')
                st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

# ====================
# TAB 4: RESULTADOS
# ====================

with tab4:
    st.header("Resultados e Perfis")
    
    if st.session_state.interpretacoes is None:
        st.warning("Execute o clustering na aba 'Clustering' primeiro")
    else:
        df_features = st.session_state.df_features
        interpretacoes = st.session_state.interpretacoes
        perfis_df = st.session_state.perfis_df
        k_kmeans = st.session_state.k_kmeans
        
        st.subheader("Perfis Identificados")
        
        for cluster_id in range(k_kmeans):
            perfil = perfis_df[perfis_df['cluster'] == cluster_id].iloc[0]
            interp = interpretacoes[cluster_id]
            
            with st.expander(f"**{interp['nome']}** ({interp['tamanho']} atletas)", expanded=(cluster_id==0)):
                st.write(f"**Descrição:** {interp['descricao']}")
                st.write(f"**Posições:** {interp['posicoes_sugeridas']}")
                st.write(f"**Treinamento:** {interp['foco_treinamento']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Idade", f"{perfil['idade_media']:.1f}")
                    st.metric("Altura", f"{perfil['altura_media']:.1f}")
                with col2:
                    st.metric("Força", f"{perfil['forca_media']:.1f}")
                    st.metric("Velocidade", f"{perfil['velocidade_media']:.2f}")
                with col3:
                    st.metric("Reflexo", f"{perfil['reflexo_medio']:.1f}")
                    st.metric("Precisão", f"{perfil['precisao_media']:.1f}")
        
        st.markdown("---")
        st.subheader("Visualizações")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df_features, x='Indice_Atletico', y='Indice_Tecnico',
                           color='Cluster_KMeans', hover_data=['Nome', 'Idade'],
                           title='Índice Atlético vs Técnico')
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        with col2:
            fig = px.scatter(df_features, x='Altura', y='Velocidade',
                           color='Cluster_KMeans', hover_data=['Nome', 'Idade'],
                           title='Altura vs Velocidade')
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        st.markdown("---")
        st.subheader("Gráfico Radar - Comparação de Perfis")
        
        # Preparar dados para o radar chart
        categories = ['Altura', 'Velocidade', 'Força', 'Reflexo', 'Precisão']
        
        fig = go.Figure()
        
        for cluster_id in range(k_kmeans):
            cluster_data = df_features[df_features['Cluster_KMeans'] == cluster_id]
            values = []
            
            for cat in categories:
                if cat == 'Velocidade':
                    # Para velocidade, menor é melhor, então invertemos
                    normalized = 100 - ((cluster_data[cat].mean() - df_features[cat].min()) / 
                                      (df_features[cat].max() - df_features[cat].min()) * 100)
                else:
                    normalized = ((cluster_data[cat].mean() - df_features[cat].min()) / 
                                (df_features[cat].max() - df_features[cat].min()) * 100)
                values.append(normalized)
            
            nome_cluster = interpretacoes[cluster_id]['nome']
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f'{nome_cluster}',
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title='Perfil dos Clusters - Gráfico Radar<br><sub>(Valores normalizados 0-100)</sub>',
            height=600
        )
        
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        st.info("""
        **Como interpretar o gráfico radar:**
        - Valores próximos a **100** indicam desempenho superior naquela característica
        - A **área coberta** por cada perfil mostra seu equilíbrio geral
        - Perfis **pontiagudos** indicam especialização em certas características
        - Perfis **circulares** indicam atletas mais equilibrados
        """)
        
        st.markdown("---")
        st.subheader("Análise por Faixa Etária")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição de clusters por idade
            idade_cluster = df_features.groupby(['Idade', 'Cluster_KMeans']).size().reset_index(name='count')
            fig = px.bar(idade_cluster, x='Idade', y='count', color='Cluster_KMeans',
                        title='Distribuição de Clusters por Idade',
                        labels={'count': 'Número de Atletas', 'Cluster_KMeans': 'Cluster'},
                        barmode='stack')
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
        with col2:
            # Boxplot altura por cluster
            fig = px.box(df_features, x='Cluster_KMeans', y='Altura', color='Cluster_KMeans',
                        title='Distribuição de Altura por Cluster',
                        labels={'Cluster_KMeans': 'Cluster', 'Altura': 'Altura (cm)'})
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot índices
            fig = px.scatter(df_features, x='Indice_Atletico', y='Indice_Tecnico',
                           color='Cluster_KMeans', size='Idade',
                           hover_data=['Nome', 'Idade', 'Altura'],
                           title='Índice Atlético vs Técnico por Cluster',
                           labels={'Indice_Atletico': 'Índice Atlético', 
                                  'Indice_Tecnico': 'Índice Técnico',
                                  'Cluster_KMeans': 'Cluster'})
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        with col2:
            # Boxplot potencial físico por cluster
            fig = px.box(df_features, x='Cluster_KMeans', y='Potencial_Fisico', color='Cluster_KMeans',
                        title='Potencial Físico por Cluster',
                        labels={'Cluster_KMeans': 'Cluster', 'Potencial_Fisico': 'Potencial Físico'})
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        st.markdown("---")
        st.subheader("Tabela Completa")
        
        display_df = df_features[[
            'Nome', 'Idade', 'Altura', 'Velocidade', 'Força', 'Reflexo', 'Precisão',
            'Cluster_KMeans', 'Indice_Atletico', 'Indice_Tecnico'
        ]].copy()
        
        display_df['Perfil'] = display_df['Cluster_KMeans'].map(lambda x: interpretacoes[x]['nome'])
        
        st.dataframe(display_df, width='stretch', height=500)

# ====================
# TAB 5: DOWNLOAD
# ====================

with tab5:
    st.header("Download de Relatórios")
    
    if st.session_state.interpretacoes is None:
        st.warning("Complete a análise primeiro")
    else:
        df_features = st.session_state.df_features
        interpretacoes = st.session_state.interpretacoes
        
        relatorio_df = df_features.copy()
        relatorio_df['Perfil'] = relatorio_df['Cluster_KMeans'].map(lambda x: interpretacoes[x]['nome'])
        relatorio_df['Descricao'] = relatorio_df['Cluster_KMeans'].map(lambda x: interpretacoes[x]['descricao'])
        relatorio_df['Posicoes'] = relatorio_df['Cluster_KMeans'].map(lambda x: interpretacoes[x]['posicoes_sugeridas'])
        relatorio_df['Treinamento'] = relatorio_df['Cluster_KMeans'].map(lambda x: interpretacoes[x]['foco_treinamento'])
        
        st.subheader("Preview")
        st.dataframe(relatorio_df.head(20), width='stretch')
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = relatorio_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "Download Relatório Completo",
                csv,
                f"relatorio_clustering_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                width='stretch'
            )
        
        with col2:
            resumo = pd.DataFrame([
                {
                    'Cluster': k,
                    'Perfil': v['nome'],
                    'Quantidade': v['tamanho'],
                    'Descricao': v['descricao']
                }
                for k, v in interpretacoes.items()
            ])
            
            csv_resumo = resumo.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "Download Resumo dos Perfis",
                csv_resumo,
                f"resumo_perfis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                width='stretch'
            )

# Footer
st.markdown("---")
st.markdown("**Sistema de Análise de Clusterização | 2025**", unsafe_allow_html=True)