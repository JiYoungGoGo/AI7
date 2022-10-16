#!/usr/bin/env python
# coding: utf-8

# In[321]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


import koreanize_matplotlib

# 그래프에 retina display 적용
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

pd.Series([1,-1]).plot(title="한글")


# In[483]:


np.random.default_rng(42)
sample_no = np.random.choice(df['가입자 일련번호'].unique(), 1000)


# In[9]:


from glob import glob

file_name = glob("성*")[0]
file_name


# In[245]:


pd.options.display.max_columns = None


# In[29]:


인구df =  pd.read_excel(file_name)
인구df.shape


# In[30]:


인구df = 인구df.drop(columns=["시나리오별(1)"])
인구df


# In[31]:


인구df = 인구df.rename(columns={"시도별(1)":"시도명", "성별(1)" : "성별", "연령별(1)" : "연령대" ,"2020":"인구"})
인구df


# In[54]:


인구df["성별"] = 인구df["성별"].replace("계", "전체")
인구df


# In[287]:


인구df.info()


# In[112]:


전국인구df =  인구df.loc[(인구df["시도명"] == "전국") & (인구df["성별"] == "전체")]
전국인구df


# In[113]:


전국인구df = 전국인구df.drop(columns=["성별", "시도명"])


# In[114]:


전국인구df["연령대"] = 전국인구df["연령대"].str.replace(" - ", "~")
전국인구df["연령대"] 


# In[115]:


전국인구df.iloc[-4:]["인구"].sum()


# In[116]:


전국인구df = 전국인구df.append({"연령대":"85세+", "인구":전국인구df.iloc[-4:]["인구"].sum()}, ignore_index=True)
전국인구df


# In[117]:


전국인구df["연령대"] = 전국인구df["연령대"].replace({"0~4세":"00~04세", "5~9세" : "05~09세"})
전국인구df


# In[498]:


sample_file = glob("sampl*.csv")[0]
df = pd.read_csv(sample_file)
df.head()


# In[ ]:





# In[481]:


df.shape


# In[71]:


df.sample(3)


# In[121]:


연령별처방df = df.groupby("연령대")[["처방내역일련번호"]].count().merge(전국인구df, on="연령대")
연령별처방df


# In[288]:


연령별처방df.describe()


# In[289]:


연령별처방df.describe(include="object")


# In[125]:


연령별처방df.plot(x="연령대", y="인구")


# In[154]:


연령별처방df["처방비율"] = 연령별처방df["처방내역일련번호"] / 연령별처방df["인구"] * 100


# In[325]:


연령별처방df.hist(bins=20, figsize=(12,10))


# In[155]:


연령별처방df.sort_values("연령대").plot(x="연령대", y="처방비율")


# In[156]:


plt.figure(figsize=(20, 4))
sns.lineplot(data=연령별처방df, x="연령대", y="처방비율")


# In[285]:


plt.figure(figsize=(20, 4))
sns.barplot(data=연령별처방df, x="연령대", y="인구", ci=None)


# In[157]:


plt.figure(figsize=(20, 4))
sns.barplot(data=연령별처방df, x="연령대", y="처방비율", ci=None)


# In[328]:


px.histogram(연령별처방df, x="연령대", y="처방비율")


# In[244]:


plt.figure(figsize=(20, 4))
sns.barplot(data=연령별처방df[4:-1], x="연령대", y="처방비율", ci=None)


# In[128]:


plt.figure(figsize=(20, 4))
sns.countplot(data=df.sort_values("연령대"), x="연령대")


# In[196]:


전국성별df =  인구df.loc[(인구df["시도명"] =="전국") & (인구df["성별"] != "전체")]
전국성별df.sample(5)


# In[197]:


전국성별df =  전국성별df.drop(columns="시도명")


# In[198]:


전국성별df["연령대"] = 전국성별df["연령대"].str.replace(" - ", "~")
전국성별df["연령대"] 


# In[203]:


남자85 = 전국성별df.loc[전국성별df["성별"] == "남자"].iloc[-4:]["인구"].sum()


# In[206]:


전국성별df = 전국성별df.append({"성별":"남자","연령대":"85세+", "인구": 남자85}, ignore_index=True)
전국성별df


# In[210]:


여자85 = 전국성별df.loc[전국성별df["성별"] == "여자"].iloc[-4:]["인구"].sum()


# In[217]:


전국성별df = 전국성별df.append({"성별":"여자", "연령대":"85세+", "인구": 여자85}, ignore_index=True)
전국성별df


# In[221]:


전국성별df["연령대"] = 전국성별df["연령대"].replace({"0~4세":"00~04세", "5~9세" : "05~09세"})
전국성별df


# In[312]:


전국인구df


# In[320]:


전국성별비율df =  전국성별df.merge(전국인구df, on="연령대")
전국성별비율df = 전국성별비율df.rename(columns={"인구_x":"인구", "인구_y":"전체인구"})
전국성별비율df["성별비"] = 전국성별비율df["인구"] / 전국성별비율df["전체인구"]
전국성별비율df


# In[165]:


df.sample()


# In[219]:


df.groupby(["연령대", "성별"])[["처방내역일련번호"]].count()


# In[222]:


전국성별df.head()


# In[291]:


전국나이성별df =  df.groupby(["연령대", "성별"])[["처방내역일련번호"]].count().merge(전국성별df, on=["연령대", "성별"])
전국나이성별df


# In[333]:


전국나이성별df = 전국나이성별df.rename(columns={"처방내역일련번호":"처방수"})


# In[334]:


전국나이성별df["인구"] = 전국나이성별df["인구"].astype(int)


# In[335]:


전국나이성별df["처방비율"] = 전국나이성별df["처방수"] / 전국나이성별df["인구"] * 100
전국나이성별df


# In[248]:


전국나이성별df.plot(kind="bar", x="연령대", y="처방비율")


# In[331]:


전국나이성별df


# In[479]:


plt.figure(figsize=(20, 4))
sns.barplot(data=연령별처방df, x="연령대", y="인구", ci=None)


# In[478]:


plt.figure(figsize=(20, 4))
sns.barplot(data=전국나이성별df, x="연령대", y="인구", hue="성별", ci=None)


# In[337]:


px.bar(전국나이성별df, x="연령대", y="처방비율", facet_col="성별")


# In[339]:


px.bar(전국나이성별df, x="연령대", y="처방비율", color="성별" , barmode="group")


# In[336]:


plt.figure(figsize=(20, 4))
sns.barplot(data=전국나이성별df, x="연령대", hue="성별", y="처방비율", ci=None)


# In[187]:


plt.figure(figsize=(20, 4))
sns.countplot(data=df, x="연령대", hue="성별", order=sorted(df["연령대"].unique())) 


# In[246]:


df.sample()


# In[265]:


금액나이성별df = df.groupby(["연령대", "성별"])["금액"].agg(["sum", "mean", "median"]).merge(전국성별df, on=["연령대", "성별"])
금액나이성별df.columns


# In[266]:


금액나이성별df.head()


# In[268]:


금액나이성별df = 금액나이성별df.rename(columns={"sum":"총금액", "mean" : "평균금액", "median":"중간금액"})
금액나이성별df.head()


# In[277]:


금액나이성별df["인구"] = 금액나이성별df["인구"].astype(int)


# In[279]:


금액나이성별df["총금액비율"] = 금액나이성별df["총금액"]/금액나이성별df["인구"] * 100


# In[280]:


금액나이성별df["평균금액비율"] = 금액나이성별df["평균금액"]/금액나이성별df["인구"] * 100


# In[340]:


금액나이성별df.head()


# In[341]:


금액나이성별df.corr()


# In[273]:


df.pivot_table(index=["연령대", "성별"], values="금액", aggfunc="sum").unstack()


# In[301]:


df.pivot_table(index=["연령대", "성별"], values="금액", aggfunc="sum").unstack().plot.bar()


# In[307]:


df.pivot_table(index=["연령대", "성별"], values="금액", aggfunc="median").unstack().plot.bar(figsize=(20, 5), rot=0)


# In[298]:


plt.figure(figsize=(20, 5))
sns.barplot(data=금액나이성별df, x="연령대", hue="성별", y="중간금액")


# In[309]:


plt.figure(figsize=(20, 5))
sns.barplot(data=금액나이성별df, x="연령대", hue="성별", y="총금액")


# In[308]:


df.pivot_table(index=["연령대", "성별"], values="금액", aggfunc="sum").unstack().plot.bar(figsize=(20, 5), rot=0)


# In[310]:


plt.figure(figsize=(20, 6))
sns.barplot(data=금액나이성별df, x="연령대", hue="성별", y="총금액비율")


# In[345]:


px.histogram(금액나이성별df, x="연령대", y="총금액")


# In[348]:


px.histogram(금액나이성별df, x="연령대", y="총금액",color="성별", barmode="group")


# In[347]:


px.histogram(금액나이성별df, x="연령대", y="총금액비율",color="성별", barmode="group")


# In[306]:


df.pivot_table(index=["연령대", "성별"], values="금액", aggfunc="mean").unstack().plot.bar(figsize=(20, 5), rot=0)


# In[353]:


plt.figure(figsize=(20, 5))
sns.barplot(data=금액나이성별df, x="연령대", hue="성별", y="평균금액비율")


# In[354]:


df.sample(2)


# In[361]:


투약df = df[["연령대", "성별", "1일투약량", "1회 투약량", "총투여일수"]]


# In[362]:


투약df["하루총투약량"] =  투약df["1일투약량"] * 투약df["1회 투약량"]


# In[363]:


투약df["총투약량"] =  투약df["하루총투약량"] * 투약df["총투여일수"]


# In[370]:


투약df


# In[373]:


투약df.plot(x="연령대")


# In[381]:


plt.figure(figsize=(20, 5))
sns.barplot(data=투약df, x="연령대", y="하루총투약량", ci=None, order= age_dict.values())


# In[383]:


plt.figure(figsize=(20, 5))
sns.barplot(data=투약df, x="연령대", y="총투여일수", ci=None, order= age_dict.values())


# In[384]:


plt.figure(figsize=(20, 5))
sns.barplot(data=투약df, x="연령대", y="총투약량", ci=None, order= age_dict.values())


# In[385]:


df.columns


# In[390]:


금액df =  df[["연령대", "성별", "총투여일수", "단가", "금액", "투여코드", "제형"]]
금액df


# In[391]:


금액df.corr()


# In[392]:


금액df.hist()


# In[418]:


제형14 = 금액df["제형"].value_counts().nlargest(14)
제형14
# 제형14.index


# In[409]:


제형14df = 금액df.loc[금액df["제형"].isin(제형14.index)]


# In[410]:


plt.figure(figsize=(20, 5))
sns.barplot(data=제형14df , x="제형", y="단가", ci=None)


# In[419]:


plt.figure(figsize=(20, 5))
sns.barplot(data=제형14df , x="제형", y="금액", ci=None)


# In[423]:


plt.figure(figsize=(20, 5))
sns.barplot(data=금액df , x="연령대", y="단가", ci=None, order= sorted(금액df["연령대"].unique()))


# In[424]:


plt.figure(figsize=(20, 5))
sns.barplot(data=금액df , x="연령대", y="금액", ci=None, order= sorted(금액df["연령대"].unique()))


# In[434]:


plt.figure(figsize=(20, 5))
sns.boxplot(data=금액df, y="단가")


# In[441]:


q1 = 금액df["단가"].quantile(0.25)
q3 = 금액df["단가"].quantile(0.75)
iqr = q3 - q1
upper = iqr * 1.5 + q3
lower = q1 - iqr * 1.5 
lower


# In[445]:


금액df[금액df["단가"] > upper]


# In[446]:


이상치제거df = 금액df.drop(금액df[금액df["단가"] > upper].index)
이상치제거df.shape


# In[449]:


plt.figure(figsize=(20, 5))
sns.boxplot(data=금액df, x="투여코드", y="단가")


# In[448]:


plt.figure(figsize=(20, 5))
sns.boxplot(data=이상치제거df, x="투여코드", y="단가")


# In[469]:


금액df["투여코드"] = 금액df["투여코드"].str.strip()
금액df["투여코드"].value_counts().index


# In[473]:


주사제df = 금액df[금액df["투여코드"]=="주사제"]
주사제df


# ## EDA 결과 알게된 것
# ### 처방 횟수 자체는 60-64 세가 가장 많지만, 인구수 대비 비율로 봤을 때 처방비율은 80-84세가 가장 많다. 
# ### 처방 횟수로 봤을 때 노령에도 여성들이 처방을 많이 받는 것으로 나타난다. 하지만 인구 비율로 봤을때 75세 이상에서는 남자들의 처방비율이 더 높은 것을 알 수 있다. 
# ### 59세까지는 남자 인구가 더 많고, 60세 이상부터는 여성의 인구비율이 더 높은 것을 알 수 있다. 
# ### 처방정보만 봤을 때는 60-64세의 남자의 총 처방금액이 높았는데, 인구대비로 봤을 때는 80-84세 남여의 처방금액비율이 가장 높다. 
# ### 0~4세 아이들의 약 처방 금액단가는 다른 연령대에 비해 3배정도 낮다.
