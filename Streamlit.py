#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 
from xgboost.sklearn import XGBClassifier
#Colocamos el lugar de donde extraer el csv
data_location= "C:/Users/LX569DW/Downloads/TP Digital House/england-premier-league-players-2018-to-2019-stats.csv"

#Leemos el csv con lo jugadores como indice
df= pd.read_csv(data_location)

#Comenzamos con la creación de la variable target, cuyo valor serán los mejores 15 jugadores de cada posición

df['top_mid_15']=df.rank_in_league_top_midfielders.apply(lambda x: 1 if x>0 and x<=15 else 0)
df['top_def_15']=df.rank_in_league_top_defenders.apply(lambda x: 1 if x>0 and x<=15 else 0)
df['top_att_15']=df.rank_in_league_top_attackers.apply(lambda x: 1 if x>0 and x<=15 else 0)
df['top_15']= df.apply(lambda x: 1 if x.top_mid_15==1 or x.top_def_15==1 or x.top_att_15==1 else 0,axis=1)
df=df.drop(columns=['top_mid_15', 'top_att_15', 'top_def_15'])

df['top_rank']=df.loc[:,['rank_in_league_top_attackers','rank_in_league_top_midfielders','rank_in_league_top_defenders']].apply(lambda x: x.min(),axis=1)


categorical_columns=['position','Current Club','nationality']
for column in categorical_columns:
    dummies = pd.get_dummies(df[column], prefix=column,drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=column)

X=df.loc[:,['goals_per_90_overall', 'assists_per_90_overall', 'goals_involved_per_90_overall', 'min_per_conceded_overall', 'minutes_played_overall']]
y= df.top_15

from sklearn.model_selection import train_test_split

#Con estratificación en y

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=162)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)

xgb = XGBClassifier(subsample= 0.8999999999999999,n_estimators=100, max_depth=6, learning_rate= 0.1,colsample_bytree= 0.6, colsample_bylevel= 0.7999999999999999)
xgb.fit(X_train, y_train)


def welcome(): 
    return 'welcome all'
  
def prediction(goals_per_90_overall, assists_per_90_overall, goals_involved_per_90_overall, min_per_conceded_overall, minutes_played_overall):   
    mw=np.array([goals_per_90_overall, assists_per_90_overall, goals_involved_per_90_overall, min_per_conceded_overall, minutes_played_overall]).reshape(1,-1)
    prediction = xgb.predict(mw)
    print(prediction) 
    return prediction 
  
def main(): 
      
    st.title("Top 15 Prediction") 
    html_temp = ""
    
    st.markdown(html_temp, unsafe_allow_html = True) 
    goals_per_90_overall_input = st.number_input("goals per 90 overall") 
    assists_per_90_overall_input = st.number_input("assists per 90 overall") 
    goals_involved_per_90_overall_input = st.number_input("goals_involved_per_90_overall") 
    min_per_conceded_overall_input = st.number_input("min_per_conceded_overall") 
    minutes_played_overall_input= st.number_input("minutes_played_overall") 
    result ="" 
      
    
    
    if st.button("Predict"): 
        result = prediction(goals_per_90_overall_input,assists_per_90_overall_input,goals_involved_per_90_overall_input,goals_involved_per_90_overall_input,min_per_conceded_overall_input)
        if result==1:
            result='top 15 in his position'
        else:
            result= 'not top 15 in his position'
            
    st.success('The player is {}'.format(result)) 
if __name__=='__main__': 
    main() 
    


# In[ ]:





# In[ ]:





# In[ ]:




