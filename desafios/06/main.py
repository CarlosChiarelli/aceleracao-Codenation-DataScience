#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)
#import sklearn as sk


# In[4]:


# # Algumas configurações para o matplotlib.
# #%matplotlib inline
# from IPython.core.pylabtools import figsize
# figsize(12, 8)
# sns.set()


# In[5]:


countries = pd.read_csv("countries.csv")


# In[6]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# In[7]:


countries['Pop_density'].str.replace(',','.')


# In[8]:


cols_transf_Num = ['Birthrate','Deathrate','Agriculture',
                   'Industry','Service','Literacy','Phones_per_1000',
                   'Arable','Crops','Other','Pop_density','Coastline_ratio',
                   'Net_migration','Infant_mortality']

for coluna in cols_transf_Num:
    countries[coluna] = countries[coluna].str.replace(',','.').astype(float)


# In[9]:


countries.dtypes


# In[10]:


cols_strip = ['Country','Region']

for coluna in cols_strip:
    countries[coluna] = countries[coluna].str.strip()


# In[11]:


countries.head()


# In[12]:


countries['Region'].unique()


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[13]:


# Sua análise começa aqui.


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[14]:


def q1():
    return list(np.sort(countries['Region'].unique()))


# In[15]:


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[16]:


countries.head()


# In[17]:


from sklearn.preprocessing import KBinsDiscretizer

discretizador = KBinsDiscretizer(n_bins=10,
                                 encode='ordinal',
                                 strategy='quantile')

discretizador.fit(countries[['Pop_density']])

popDensity_disc = discretizador.transform(countries[['Pop_density']])

popDensity_disc


# In[18]:


np.unique(popDensity_disc.flatten())


# In[19]:


pd.Series(popDensity_disc.flatten()).value_counts()[9.0]


# In[20]:


def q2():
    # Retorne aqui o resultado da questão 2.
    from sklearn.preprocessing import KBinsDiscretizer

    discretizador = KBinsDiscretizer(n_bins=10,
                                     encode='ordinal',
                                     strategy='quantile')

    discretizador.fit(countries[['Pop_density']])

    popDensity_disc = discretizador.transform(countries[['Pop_density']])

    return int(pd.Series(popDensity_disc.flatten()).value_counts()[9.0])


# In[21]:


q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[23]:


def q3():
    #Hot enconding para np.int
    from sklearn.preprocessing import OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse=False)
    
    #Codificando as variáveis
    region_climate_encoded = one_hot_encoder.fit(countries[['Region', 'Climate']].fillna('0').astype('str'))
    
    #Pegando as novas features geradas
    new_attributes = region_climate_encoded.get_feature_names()
    
    return len(new_attributes)
q3()


# In[24]:


q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[25]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]

test_df = pd.DataFrame(data=np.array(test_country).reshape(1,-1), columns=countries.columns)


# In[26]:


cols_num = (countries.dtypes=='int64') | (countries.dtypes=='float64')
cols_num = countries.dtypes[cols_num].index.tolist()
countries[cols_num]


# In[27]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline(steps=[
    ('imputacao', SimpleImputer(strategy='median')),
    ('padronizacao', StandardScaler())
])

num_pipeline = num_pipeline.fit(countries[cols_num])

test_country_transf = num_pipeline.transform(test_df[cols_num])

test_country_transf = pd.DataFrame(test_country_transf.flatten().reshape(1,-1),
                                   columns=cols_num)
valor = np.round(test_country_transf['Arable'], 3)
float(valor)


# In[29]:


def q4():
    # Retorne aqui o resultado da questão 4.
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    num_pipeline = Pipeline(steps=[
        ('imputacao', SimpleImputer(strategy='median')),
        ('padronizacao', StandardScaler())
    ])

    num_pipeline = num_pipeline.fit(countries[cols_num])

    test_country_transf = num_pipeline.transform(test_df[cols_num])

    test_country_transf = pd.DataFrame(test_country_transf.flatten().reshape(1,-1),
                                       columns=cols_num)
    valor = np.round(test_country_transf['Arable'], 3)
    return float(valor)


# In[30]:


q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[31]:


def q5():

    quartil_01 = countries['Net_migration'].quantile(.25)
    quartil_03 = countries['Net_migration'].quantile(.75)

    iqr = quartil_03 - quartil_01
    lim_inf, lim_sup = (quartil_01 - 1.5*iqr), (quartil_03 + 1.5*iqr)

    outl_inf = countries['Net_migration'] < lim_inf
    outl_inf = countries[outl_inf]['Net_migration']

    outl_sup = countries['Net_migration'] > lim_sup
    outl_sup = countries[outl_sup]['Net_migration']

    dados_inf_pad = StandardScaler().fit_transform(countries[['Net_migration']])
    dados_inf_pad = dados_inf_pad[countries['Net_migration'] < lim_inf]
    quant_outl_inf = (dados_inf_pad < 3).sum()

    dados_sup_pad = StandardScaler().fit_transform(countries[['Net_migration']])
    dados_sup_pad = dados_sup_pad[countries['Net_migration'] > lim_sup]
    quant_outl_sup = (dados_sup_pad > 3).sum()

    if (quant_outl_inf + quant_outl_sup) > 1:
        remover = True
    else:
        remover = False

    return (len(outl_inf), len(outl_sup), bool(0))


# In[32]:


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[40]:


def q6():
    # Retorne aqui o resultado da questão 4.
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)

    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroup.data)

    aux = pd.DataFrame(newsgroups_counts.toarray(),
                 columns=np.array(count_vectorizer.get_feature_names()))

    return int(aux['phone'].sum())


# In[41]:


q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[ ]:





# In[42]:


def q7():
    # Retorne aqui o resultado da questão 4.
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)

    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroup.data)

    aux = pd.DataFrame(newsgroups_counts.toarray(),
                 columns=np.array(count_vectorizer.get_feature_names()))
    
    tfidf_transformer = TfidfTransformer()

    tfidf_transformer.fit(newsgroups_counts)

    newsgroups_tfidf = tfidf_transformer.transform(newsgroups_counts)

    aux = pd.DataFrame(newsgroups_tfidf.toarray(),
                 columns=np.array(count_vectorizer.get_feature_names()))

    return float(round(aux['phone'].sum(), 3))


# In[43]:


q7()


# In[ ]:





# In[ ]:





# In[ ]:




