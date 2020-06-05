#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[62]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[63]:


#%matplotlib inline

from IPython.core.pylabtools import figsize

figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[64]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[4]:


# Sua análise da parte 1 começa aqui.


# In[65]:


dataframe.head()


# In[66]:


dataframe.shape


# In[67]:


dataframe.info()


# In[68]:


dataframe.describe()


# In[69]:


sns.distplot(dataframe.normal)


# In[70]:


sns.distplot(dataframe.binomial, hist_kws={'alpha': .5})


# In[71]:


sns.distplot(dataframe.normal, hist_kws={'alpha': .5})
sns.distplot(dataframe.binomial, hist_kws={'alpha': .5})


# In[ ]:





# In[ ]:





# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[72]:


def q1():
    # Retorne aqui o resultado da questão 1.
    aux = dataframe.describe().loc['25%':'75%']
    aux = list(aux.normal - aux.binomial)
    aux = [round(x, 3) for x in aux]
    return tuple(aux)
    


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# Sim.
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?
# 
# A binomial pode ser aproximada para uma normal, logo é plausível quartis próximos já que possuem a mesma média e variância.

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[ ]:





# In[140]:


def q2():
    # Retorne aqui o resultado da questão 2.
    
#     estat = dataframe['normal'].agg(['mean','std'])
#     limInfer = estat['mean'] - estat['std']
#     limSup = estat['mean'] + estat['std']

#     # cauda inferior
#     # 1 - P(x >= x-s) 
#     probInf = 1 - sct.norm.sf(limInfer, loc=estat['mean'], scale=estat['std'])

#     # cauda superior
#     # 1 - P(x <= x+s)
#     probSup = 1 - sct.norm.cdf(limSup, loc=estat['mean'], scale=estat['std'])
   
    # intervalo
#     prob = 1 - (probInf + probSup)

#     float(round(prob, 3))
    
    media_normal = dataframe['normal'].mean()
    desvio_padrao_normal = dataframe['normal'].std()
    prob = ECDF(dataframe['normal'])
    resposta = float(round(prob(media_normal + desvio_padrao_normal), 3) - round(prob(media_normal - desvio_padrao_normal),3))

    return resposta
    


# In[141]:


q2()


# In[139]:


q2()


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# 
# Sim
# 
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[76]:


def q3():
    # Retorne aqui o resultado da questão 3.
    aux = dataframe.agg({'mean','var'})
    aux = aux['binomial'] - aux['normal']
    aux = aux.loc[['mean','var']]
    return tuple([round(x, 3) for x in aux])
    


# In[77]:


q3()


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# 
# Pequena magnitude.
# 
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[78]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[9]:


# Sua análise da parte 2 começa aqui.


# In[79]:


stars.head()


# In[80]:


stars.shape


# In[82]:


stars.info()


# In[ ]:





# In[ ]:





# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[124]:


filtro = stars['target'] == False
aux = stars[filtro]
# padronizando
aux['false_pulsar_mean_profile_standardized'] = (aux.mean_profile - aux.mean_profile.mean())/aux.mean_profile.std()

media = aux['false_pulsar_mean_profile_standardized'].mean()
desvPad = aux['false_pulsar_mean_profile_standardized'].std()

# quartis
quartis = [sct.norm.ppf(x, loc=0, scale=1) for x in (.8, .9, .95)]

# prob acumulada
[round(sct.norm.cdf(x, loc=media, scale=desvPad), 3) for x in quartis]


# In[121]:


print(media,desvPad)


# In[94]:


sns.distplot(aux.false_pulsar_mean_profile_standardized)


# In[134]:


def q4():
    # Retorne aqui o resultado da questão 4.
    filtro = stars['target'] == False
    aux = stars[filtro]
    # padronizando
    aux['false_pulsar_mean_profile_standardized'] = (aux.mean_profile - aux.mean_profile.mean())/aux.mean_profile.std()

    media = aux['false_pulsar_mean_profile_standardized'].mean()
    desvPad = aux['false_pulsar_mean_profile_standardized'].std()

    # quartis
    quartis = [sct.norm.ppf(x, loc=0, scale=1) for x in (.8, .9, .95)]
    
    # prob acumulada
    return tuple([round(sct.norm.cdf(x, loc=media, scale=desvPad), 3) for x in quartis])


# In[135]:


q4()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# 
# Sim, pois apliquei a reversa da função.
# 
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# 
# Que ela obedece uma distribuição normal.

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[108]:


filtro = stars['target'] == False
aux = stars[filtro]
# padronizando
aux['false_pulsar_mean_profile_standardized'] = (aux.mean_profile - aux.mean_profile.mean())/aux.mean_profile.std()

#aux = aux[['mean_profile','false_pulsar_mean_profile_standardized']]
quartis = aux.false_pulsar_mean_profile_standardized.describe().loc['25%':'75%']


# In[112]:


quartisPad = list(sct.norm.ppf(x, loc=0, scale=1) for x in (.25, .5, .75))
tuple(round(quartis-quartisPad, 3))


# In[ ]:





# In[11]:


def q5():
    # Retorne aqui o resultado da questão 5.
    filtro = stars['target'] == False
    aux = stars[filtro]
    # padronizando
    aux['false_pulsar_mean_profile_standardized'] = (aux.mean_profile - aux.mean_profile.mean())/aux.mean_profile.std()

    #aux = aux[['mean_profile','false_pulsar_mean_profile_standardized']]
    quartis = aux.false_pulsar_mean_profile_standardized.describe().loc['25%':'75%']
    
    quartisPad = list(sct.norm.ppf(x, loc=0, scale=1) for x in (.25, .5, .75))
    return tuple(round(quartis-quartisPad, 3))


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# 
# Fazem.
# 
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# 
# Ela obedece uma distribuição normal.
# 
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.

# In[ ]:




