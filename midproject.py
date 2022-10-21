from cmath import log
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np

st.title("전기차충전소는 어디에 얼마나 생겨야 할까?")

st.markdown("## 데이터")
st.markdown("서울시총인구, 서울시전기차, 서울시충전소, 서울시휘발유차, 서울시주유소 데이터를 합친것!!")
file_name = ("data/midproject.csv")
df = pd.read_csv(file_name, encoding='cp949')
df

st.markdown("## 구별 전기차 비율")
pie = px.pie(df, values='전기차수', names='구', width=1200, height=700)
st.plotly_chart(pie)

st.markdown("## 미래전기차수-휘발유자동차수-전기차수")
ptx = px.line(df,  x="구", y=['미래전기차수', "휘발유자동차수", "전기차수"])
st.plotly_chart(ptx)


st.markdown("## 상관계수")
fig, ax = plt.subplots(figsize=(10, 3))
mask = np.triu(np.ones_like(df[["인구", "주유소개수", "전기차수", "충전소개수", "휘발유자동차수", "미래전기차수"]].corr()))
sns.heatmap(df[["인구", "주유소개수", "전기차수", "충전소개수", "휘발유자동차수", "미래전기차수"]].corr(), annot=True, fmt=".2f", cmap="Greens", mask=mask)
st.pyplot(fig)

st.markdown("## 인구")
ptx = px.bar(df, y="인구", x="구")
ptx

st.markdown("## 전기차수")
st.line_chart(data=df, x="구", y="전기차수")

pxh = px.histogram(df, x="구", y="전기차수", color='전기차수', width=800, height=400)
pxh


st.markdown("## 인구-휘발유자동차수-전기차수")
ptx = px.line(df,  x="구", y=["인구", "휘발유자동차수", "전기차수"])
ptx
ptx = px.bar(df,  x="구", y=["인구", "휘발유자동차수", "전기차수"])
ptx

st.markdown("## 휘발유자동차수-전기차수")
ptx = px.line(df,  x="구" , y=["휘발유자동차수", "전기차수"])
ptx