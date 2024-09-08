import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv("drug200.csv") # CSV yolunu değiştir dosyayı ana dizine at
onehot = OneHotEncoder(drop = 'first')
dum = onehot.fit_transform(df[['Sex', 'BP', 'Cholesterol']]).toarray()
dum_col = onehot.get_feature_names_out()

df = df.drop(columns = ['Sex', 'BP', 'Cholesterol'])
df[dum_col] = dum

y=df['Drug']
x=df.drop('Drug', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=24)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
skor = model.score(x_test, y_test)

# Dinamik Hale Getirdik (Kullanıcıdan Bilgi Alarak).
age = st.sidebar.number_input('Yaşınızı Giriniz: ')
cinsiyet = st.sidebar.selectbox('Cinsiyetinizi Giriniz F veya M: ', ['F', 'M'])
bp = st.sidebar.selectbox('Kan Basıncınız (LOW - HIGH - NORMAL): ', ['LOW', 'HIGH', 'NORMAL'])
chole = st.sidebar.selectbox('Cholesterol Seviyeniz (NORMAL - HIGH): ', ['HIGH', 'NORMAL'])
na = st.sidebar.number_input('Sodyum - Potasyum Oranınız: ')

btn = st.sidebar.button("GETİR")
if not btn:
    st.stop() # Basılmazsa Durdur

tahmin_dum = onehot.transform([[cinsiyet, bp, chole]]).toarray()
tahmin = np.array([[age, na]])
tahmin = np.append(tahmin, tahmin_dum) # Birleştirdik

sonuc = model.predict([tahmin])

st.title(sonuc[0])