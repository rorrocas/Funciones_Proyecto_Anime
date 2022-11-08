import numpy as np # Tratamiento de datos
import matplotlib.pyplot as plt # Graficos
import seaborn as sns # Graficos
import pandas as pd
import numpy as np

# Definición de estilo de graficos
plt.style.use('seaborn') # Gráficos estilo seaborn
plt.rcParams["figure.figsize"] = (6, 3) # Tamaño gráficos
plt.rcParams["figure.dpi"] = 100 # resolución gráficos
sns.set_theme()

import warnings
warnings.filterwarnings('ignore')

def Graficas_frecuencia(df):
    rows = int(np.ceil(df.shape[1] / 3))
    height = rows * 3.5
    fig = plt.figure(figsize=(12, height))
    for n, i in enumerate(df.columns):
        if df[i].dtype in ('object', 'int64') :
            fig.add_subplot(rows, 3, n+1)
            ax = sns.countplot(x=i, data=df)
            plt.title(i)
            plt.xlabel('')
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x()+p.get_width()/2., height + .5,'{:1.2f}'.format(height/len(df[i])), ha="center")
        if df[i].dtype == 'float64':
            fig.add_subplot(rows, 3, n+1)
            ax = sns.distplot(df[i])
            plt.title(i)
            plt.xlabel('')    
    plt.tight_layout()
    return 0

def calidad_datos(data):
  tipos = pd.DataFrame({'tipo': data.dtypes},index=data.columns)
  na = pd.DataFrame({'nulos': data.isna().sum()}, index=data.columns)
  na_prop = pd.DataFrame({'porc_nulos':data.isna().sum()/data.shape[0]}, index=data.columns)
  ceros = pd.DataFrame({'ceros':[data.loc[data[col]==0,col].shape[0] for col in data.columns]}, index= data.columns)
  ceros_prop = pd.DataFrame({'porc_ceros':[data.loc[data[col]==0,col].shape[0]/data.shape[0] for col in data.columns]}, index= data.columns)
  summary = data.describe(include='all').T
  summary['dist_IQR'] = summary['75%'] - summary['25%']
  summary['limit_inf'] = summary['25%'] - summary['dist_IQR']*1.5
  summary['limit_sup'] = summary['75%'] + summary['dist_IQR']*1.5
  summary['outliers'] = data.apply(lambda x: sum(np.where((x<summary['limit_inf'][x.name]) | (x>summary['limit_sup'][x.name]),1 ,0)) if x.name in summary['limit_inf'].dropna().index else 0)
  return pd.concat([tipos, na, na_prop, ceros, ceros_prop, summary], axis=1).sort_values('tipo')

def get_nombre_from_index(df_anime,index):
    return df_anime[df_anime.index == index]['name'].values[0]

def get_id_from_nombre(df_anime,name):
    return df_anime[df_anime.name == name]['anime_id'].values[0]
    
def get_index_from_id(df_anime,anime_id):
    return df_anime[df_anime.anime_id == anime_id].index.values[0]

def get_user_top_list(df_rating,user):
    df_user = df_rating[df_rating['user_id']==user]
    df_rated = df_user.dropna(how = 'any')
    avg =  df_rated.rating.mean() 
    df_toplist = df_rated[df_rated['rating']>= avg].sort_values('rating', ascending = False).head(10)
    return list(df_toplist['anime_id'])

def get_user_viewed_list(df_rating,user):
    return list(df_rating[df_rating['user_id']==user]['anime_id'])

def get_recommendations(df_anime,indices,aid):
    anime =  get_index_from_id(df_anime,aid)
    test = list(indices[anime,1:11])
    nb = []
    for i in test:
        a_name = get_nombre_from_index(df_anime,i)
        nb.append(a_name)
    return nb

def get_n_recommends(df_anime,df_rating,indices,user, n):
    vistas = list(get_user_viewed_list(df_rating,user))
    liked = list(get_user_top_list(df_rating,user))
    lista = []
    for i in liked:
        ani = pd.Series(get_recommendations(df_anime,indices,i))
        recs = np.setdiff1d(ani, vistas) 
        lista.extend(recs)
        if(len(lista) > n):
            lista = lista[n:]
            break
    return lista
