import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
from typing import Union, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyData:
    def __init__(self, csv: str, scaler = StandardScaler()):
        self.df = pd.read_csv(csv)
        self.df_evaluation = pd.DataFrame()
        self.scaler = scaler
        self.correlacao = None
        self.modulo_correlacao = None
        self.features_importantes = []
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.modelos = []

    def preparando_para_splittar(self, threshold_de_importancia:float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame]:
        mapa = {'M':1, 'B':0}
        self.df.diagnosis = self.df.diagnosis.map(mapa)

        self.df.drop(columns = 'id', inplace = True)
        self.correlacao = self.df.corr()
        self.modulo_correlacao = abs(self.correlacao)

        self.modulo_correlacao = self.modulo_correlacao[self.modulo_correlacao > threshold_de_importancia]
        self.features_importantes += list(self.modulo_correlacao.index)
        self.features_importantes.remove('diagnosis')

        self.X = self.df[self.features_importantes]
        self.y = self.df.diagnosis
        return self

    def preparando_para_evaluation(self, test_size:float = 0.20, random_state = 42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = test_size, random_state = random_state)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        return self

    def evaluate(self, model, nome) -> pd.DataFrame:
        model.fit(self.X_train, self.y_train)
        y_hat = model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_hat)
        f1 = f1_score(self.y_test, y_hat)
        precision = precision_score(self.y_test, y_hat)
        recall = recall_score(self.y_test, y_hat)
        temp = pd.DataFrame({'Accuracy':accuracy, 'F1':f1, 'Precision':precision, 'Recall':recall}, index = [nome])
        return temp

    def adicionar_modelo_ml(self, modelo_e_nome: Tuple):
        self.modelos.append({'modelo':modelo_e_nome[0], 'nome':modelo_e_nome[1]})
        return self

    def criar_dataframe_de_evaluation(self):
        for modelo in self.modelos:
            temp = self.evaluate(modelo['modelo'], modelo['nome'])
            self.df_evaluation = pd.concat([self.df_evaluation, temp])
        return self


# ======================
# Layout streamlit
# ======================

st.set_page_config(page_title = 'Analise de cancer de mama', layout = 'wide')
st.sidebar.markdown('# Analise exploratoria do dataset de cancer de mama')
st.sidebar.markdown("""---""")

threshold = st.sidebar.slider('Selecione o limite inferior para a correlacao', 0, 100, 10)
threshold /= 100

dados = MyData('breast-cancer.csv')
meus_dados_transformados = dados.preparando_para_splittar(threshold_de_importancia = threshold).preparando_para_evaluation()\
                            .adicionar_modelo_ml((LogisticRegression(), 'Regressao logistica')).adicionar_modelo_ml((DecisionTreeClassifier(), 'Arvore de decisao'))\
                            .adicionar_modelo_ml((RandomForestClassifier(), 'Random Forest')).adicionar_modelo_ml((xgb.XGBClassifier(), 'XGBoost'))\
                            .criar_dataframe_de_evaluation()

st.sidebar.markdown('## Selecione uma feature para ser analisada')
features = list(meus_dados_transformados.df.columns)
features.remove('diagnosis')

selecionador_de_feature = st.sidebar.selectbox('Escolha uma das features', features)
dataframe = st.radio('Mostrar a coluna selecionada', ['Mostrar', 'Nao mostrar'])



if dataframe == 'Mostrar':
    col1, col2 = st.columns([1, 3])
    with col1:
        with st.container():
            st.markdown('### Feature escolhida')
            st.dataframe(meus_dados_transformados.df[[selecionador_de_feature]])

    with col2:
        with st.container():
            plot = sns.displot(data = meus_dados_transformados.df, x = selecionador_de_feature, hue = 'diagnosis', kind = 'hist')
            plot.set_titles(f'Distribuicao dos dados benignos e malignos da feature {selecionador_de_feature}')
            st.pyplot(plot)

else:
    with st.container():
            plot = sns.displot(data = meus_dados_transformados.df, x = selecionador_de_feature, hue = 'diagnosis', kind = 'hist')
            plot.set_titles(f'Distribuicao dos dados benignos e malignos da feature {selecionador_de_feature}')
            st.pyplot(plot)

st.markdown('# Correlacao das variaveis')

correlacao = meus_dados_transformados.df.corr()
correlacao = abs(correlacao[correlacao > threshold])

fig, ax = plt.subplots()
corr_plot = sns.heatmap(correlacao, annot = True, ax = ax)
st.write(fig)

st.markdown('# Quao bem os modelos se sairam no dataset')
comparacao = st.multiselect('Selecione os modelos que deseja comparar', [modelos['nome'] for modelos in meus_dados_transformados.modelos], [])

if comparacao:
    st.dataframe(meus_dados_transformados.df_evaluation.loc[comparacao])
else:
    st.dataframe(meus_dados_transformados.df_evaluation)