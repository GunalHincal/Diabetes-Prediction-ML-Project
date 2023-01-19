

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import joblib
import pydotplus
import graphviz

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, SVR

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import cross_val_predict, cross_validate, GridSearchCV
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import *
from skompiler import skompile  # bize excel python ve sql formunda çıktılar verecek


pd.set_option('display.max_columns', None)  # bütün sütunları göster
pd.set_option('display.max_rows', None)     # bütün satırları göster
pd.set_option('display.width', 500)  # sütunlar max 500 tane gösterilsin
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv(r"C:\Users\GunalHincal\Desktop\projects_csv\miuul_final_project_diabetes_pred.csv")
df.head(10)
df.tail()
df.info()
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df.shape  # (70692, 18)


# df te columns isimleri irili ufaklı kolaylık olsun açısından büyütüyoruz
df.columns = [col.upper() for col in df.columns]


# check_df bi dataframe i kendisine sorduğumuzda o datanın genel resmini getiren fonksiyonumuzdur
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


def grab_col_names(dataframe, cat_th=15, car_th=25):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")  # df in satır gözlem sayısı
    print(f"Variables: {dataframe.shape[1]}")     # df in sütun değişken sayısı
    print(f'cat_cols: {len(cat_cols)}')           # categorik sütunlar
    print(f'num_cols: {len(num_cols)}')           # nümerik sütunlar
    print(f'cat_but_car: {len(cat_but_car)}')     # categorik gözüken ama kardinal olan sütunlar
    print(f'num_but_cat: {len(num_but_cat)}')     # nümerik gözüküp categorik olan sütunlar

    return cat_cols, num_cols, cat_but_car


# df veri setimizi grab_col_names fonksiyonundan geçiriyoruz
cat_cols, num_cols, cat_but_car = grab_col_names(df)


#################################
# korelasyon matrisini oluşturur, sayısal değişkenlerin birbirleriyle olan korelasyonunu hesaplar
def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


correlation_matrix(df, num_cols)


# Aykırı değerleri tespit etmek için önce eşik değerleri belirleyeceğiz
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


for col in num_cols:
    print(col, outlier_thresholds(df, col))


# hesaplanan alt-üst eşik değerlerlere göre değişkenlerde aykırı değerler var mı, bool çıktı gelir
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))


# aykırı değerleri baskılamak için fonksiyonumuz
def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, replace_with_thresholds(df, col))


# tekrar bakalım aykırı değer kalmış mı
for col in num_cols:
    print(col, check_outlier(df, col))


# GÖRSELLEŞTİRMELER
########################################################################################

# Bağımlı değişkenin sınıflarını getirir, sınıflar dengeli mi dengesiz mi gösterir
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
chd_plot = df['DIABETES'].value_counts().plot(kind='bar')
plt.xlabel('DIABETES')
plt.ylabel('COUNT')
plt.show()
########################################
pd.DataFrame(df.groupby(['AGE'])['DIABETES'].count())
# AgeGroups    Counts    AgeRange
#    1.0         979      18-24
#    2.0        1396      25-29
#    3.0        2049      30-34
#    4.0        2793      35-39
#    5.0        3520      40-44
#    6.0        4648      45-49
#    7.0        6872      50-54
#    8.0        8603      55-59
#    9.0       10112      60-64
#   10.0       10856      65-69
#   11.0        8044      70-74
#   12.0        5394      75-70
#   13.0        5426        80+
df.loc[(df["AGE"] == 1.0) & (df["DIABETES"] == 1.0)].count()   # 901 kişi=0    78 kişi=1
df.loc[(df["AGE"] == 2.0) & (df["DIABETES"] == 1.0)].count()   # 1256 kişi=0   140 kişi=1
df.loc[(df["AGE"] == 3.0) & (df["DIABETES"] == 1.0)].count()   # 1735 kişi=0   314 kişi=1
df.loc[(df["AGE"] == 4.0) & (df["DIABETES"] == 1.0)].count()   # 2167 kişi=0   626 kişi=1
df.loc[(df["AGE"] == 5.0) & (df["DIABETES"] == 1.0)].count()   # 2469 kişi=0  1051 kişi=1
df.loc[(df["AGE"] == 6.0) & (df["DIABETES"] == 1.0)].count()   # 2906 kişi=0  1742 kişi=1
df.loc[(df["AGE"] == 7.0) & (df["DIABETES"] == 1.0)].count()   # 3784 kişi=0  3088 kişi=1
df.loc[(df["AGE"] == 8.0) & (df["DIABETES"] == 1.0)].count()   # 4340 kişi=0  4263 kişi=1
df.loc[(df["AGE"] == 9.0) & (df["DIABETES"] == 1.0)].count()   # 4379 kişi=0  5733 kişi=1
df.loc[(df["AGE"] == 10.0) & (df["DIABETES"] == 1.0)].count()  # 4298 kişi=0  6558 kişi=1
df.loc[(df["AGE"] == 11.0) & (df["DIABETES"] == 1.0)].count()  # 2903 kişi=0  5141 kişi=1
df.loc[(df["AGE"] == 12.0) & (df["DIABETES"] == 1.0)].count()  # 1991 kişi=0  3403 kişi=1
df.loc[(df["AGE"] == 13.0) & (df["DIABETES"] == 1.0)].count()  # 2217 kişi=0  3209 kişi=1


# değişkenlerin sınıflarının veri içerisindeki dağılım grafiği
fig = plt.figure(figsize=(20, 15))
ax = fig.gca()
df.hist(ax=ax)
plt.show()


# 1'den 13 e yaş aralıklarını ve frekanslarını gösteren sütun ve pasta grafiği
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
plt.rcParams['figure.figsize'] = (10, 5)
ax1 = plt.subplot(1, 2, 1)
Age_status = pd.DataFrame(df.groupby(['AGE'])['DIABETES'].count())
sns.barplot(x=Age_status.index, y=Age_status['DIABETES'])
plt.xlabel('Age Groups')
plt.ylabel('Age Frekans')
plt.title('Age Distribution')
ax2 = plt.subplot(1, 2, 2)
df.groupby(['AGE'])['DIABETES'].count().plot(kind='pie')
plt.ylabel('Age Groups Frekans')
plt.title('Age Proposanate')
plt.show()


# diabetes and non-diabetes counts for each age group
Age_status = df.groupby(['AGE'])['DIABETES'].value_counts()
Age_status = Age_status.unstack()
Age_status.columns = ["Non-Diabetes", "Diabetes"]
Age_status.plot(kind='bar', stacked=True)
plt.xlabel('Age Groups')
plt.ylabel('Diabetes Frekans')
plt.title('Age Distribution')
plt.show()
Age_proportion = df.groupby(['AGE'])['DIABETES'].value_counts(normalize=True)
Age_proportion = Age_proportion.unstack()
Age_proportion.columns = ["Non-Diabetes", "Diabetes"]
Age_proportion.plot(kind='pie', subplots=True, autopct='%1.2f%%')
plt.title('Age Proportion')
plt.show()


#  yaş gruplarının diyabet oranı
plt.rcParams['figure.figsize'] = (15, 7)
df.groupby(['AGE', 'DIABETES'])['DIABETES'].count().unstack().plot(kind='bar', stacked=False)
plt.ylabel('Diyabet Frekansı')
plt.xlabel('Age Groups 1-13')
plt.title('Age Group vs Diabetes')
plt.legend(['Non-Diabetes', 'Diabetes'])
plt.show()
# bu grafikte görüyoruz ki 60 yaşa kadar kadınlarda diyabet görülme oranı daha fazla iken
# 60 yaş sonrasında farkedilir derecede erkeklerde diaybet görülme oranı artmaktadır

df.columns
# (['AGE', 'SEX', 'HIGHCHOL', 'CHOLCHECK', 'BMI', 'SMOKER', 'HEARTDISEASEORATTACK',
# 'PHYSACTIVITY', 'FRUITS', 'VEGGIES', 'HVYALCOHOLCONSUMP', 'GENHLTH', 'MENTHLTH', 'PHYSHLTH',
# 'DIFFWALK', 'STROKE', 'HIGHBP', 'DIABETES'], dtype='object')


# Cinsiyet Bazında Diyabet Frekansı
plt.rcParams['figure.figsize'] = (15, 7)
df.groupby(['SEX', 'DIABETES'])['DIABETES'].count().unstack().plot(kind='bar')
plt.ylabel('Diyabet Frekansı')
plt.xlabel('Cinsiyet (0 Kadın / 1 Erkek)')
plt.title('Cinsiyete Göre Diyabet')
plt.legend(['Non-Diabetes', 'Diabetes'])
plt.show()
# Kadın Nüfusunda Diyabet olanların frekasnısın olmayanlara göre daha fazla olduğunu görüyoruz
# Erkek Nüfusunda ise Diyabet olmayanların diyabet olanlara göre daha az olduğunu görüyoruz


# Kolesterol Seviyesi Bazında Diyabet Frekansı
plt.rcParams['figure.figsize'] = (15, 7)
df.groupby(['HIGHCHOL', 'DIABETES'])['DIABETES'].count().unstack().plot(kind='bar')
plt.ylabel('Diyabet Frekansı')
plt.xlabel('Kolesterol Durumu (0 Düşük kolesterol / 1 Yüksek kolasterol)')
plt.title('Yüksek Kolesterole Sahip olma Durumuna Göre Diyabet')
plt.legend(['Non-Diabetes', 'Diabetes'])
plt.show()
# Düşük Kolesterol seviyelerinde daha az diyabet oranı gözlemlenirken
# yüksek kolesterol seviyelerinde daha fazla diyabet oranı gözlemlenmiştir


# Kolesterol Kontrolü Yaptıranlarda ve Yaptırmayanlarda Diyabet Frekansı
plt.rcParams['figure.figsize'] = (15, 7)
df.groupby(['CHOLCHECK', 'DIABETES'])['DIABETES'].count().unstack().plot(kind='bar')
plt.ylabel('Diyabet Frekansı')
plt.xlabel('Kolesterol Kontrol Durumu (0 Düşük kolesterol / 1 Yüksek kolasterol)')
plt.title('Kolesterol Kontrolü Yaptırma Durumuna Göre Diyabet')
plt.legend(['Non-Diabetes', 'Diabetes'])
plt.show()


# BMI Bazında Diyabet Frekansı
plt.rcParams['figure.figsize'] = (15, 7)
df.groupby(['BMI', 'DIABETES'])['DIABETES'].count().unstack().plot(kind='bar')
plt.ylabel('Diyabet Frekansı')
plt.xlabel('BMI Değeri')
plt.title('BMI Durumuna Göre Diyabet')
plt.legend(['Non-Diabetes', 'Diabetes'])
plt.show()
# buradan gözlemlendiği üzere BMI arttıkça diyabet daha fazla gözlemlenmiş özellikle 28 bmi değerinden sonra
# diyabet nüfusu diyabet olmama nüfusunu geçmiş durumda bu durumda diyebiliriz ki obez olma durumu
# diyabete sahip olma konusunda önemli bir eşiktir


# Sigara Kullanıcısı olma ve olmama Bazında Diyabet Frekansı
plt.rcParams['figure.figsize'] = (15, 7)
df.groupby(['SMOKER', 'DIABETES'])['DIABETES'].count().unstack().plot(kind='bar')
plt.ylabel('Diyabet Frekansı')
plt.xlabel('SMOKER Değeri')
plt.title('Sigara içme Durumuna Göre Diyabet')
plt.legend(['Non-Diabetes', 'Diabetes'])
plt.show()
# sigara tüketme alışkanlığının da diyabet oranını etkilediği görülmektedir


# CHD Bazında Diyabet Frekansı
plt.rcParams['figure.figsize'] = (15, 7)
df.groupby(['HEARTDISEASEORATTACK', 'DIABETES'])['DIABETES'].count().unstack().plot(kind='bar')
plt.ylabel('Diyabet Frekansı')
plt.xlabel('CHD Durumu')
plt.title('Kalp Rahatsızlığı Durumuna Göre Diyabet')
plt.legend(['Non-Diabetes', 'Diabetes'])
plt.show()
# kalp rahatszılığına sahip olan insanlarda diyabet oranı daha fazla gözlemlenmiş


# Fiziksel Aktivite yapma Durumuna göre Diyabet Frekansı
plt.rcParams['figure.figsize'] = (15, 7)
df.groupby(['PHYSACTIVITY', 'DIABETES'])['DIABETES'].count().unstack().plot(kind='bar')
plt.ylabel('Diyabet Frekansı')
plt.xlabel('Fiziksel Aktivite Durumu')
plt.title('Fiziksel Aktivite Durumuna Göre Diyabet')
plt.legend(['Non-Diabetes', 'Diabetes'])
plt.show()
# fiziksel aktivite yapmayan nüfusta diyabet daha fazla gözlemlenmiş
# fiziksel aktivite yapan nüfusta ise diyabet olmayan grup daha fazla gözlemlenmiş


# Alkol Kullanımı Bazında Diyabet Frekansı
plt.rcParams['figure.figsize'] = (15, 7)
df.groupby(['HVYALCOHOLCONSUMP', 'DIABETES'])['DIABETES'].count().unstack().plot(kind='bar')
plt.ylabel('Diyabet Frekansı')
plt.xlabel('Alkol Tüketme Durumu')
plt.title('HVYALCOHOLCONSUMP Durumuna Göre Diyabet')
plt.legend(['Non-Diabetes', 'Diabetes'])
plt.show()
# alkol tüketenlerde diyabet olmama durumu daha fazla gözlemlenmiş
# tüketmeyenlerde ise diyabet daha fazla görülüyor


# Genel Sağlık Bazında Diyabet Frekansı
plt.rcParams['figure.figsize'] = (15, 7)
df.groupby(['GENHLTH', 'DIABETES'])['DIABETES'].count().unstack().plot(kind='bar')
plt.ylabel('Diyabet Frekansı')
plt.xlabel('Genel sağlık Durumu')
plt.title('Genel sağlık Durumuna Göre Diyabet (1çok iyi-2-3-4-5kötü')
plt.legend(['Non-Diabetes', 'Diabetes'])
plt.show()
# genel sağlık durumu kötüleştikçe diyabet olma frekansı atmış


#  Yürüme merdiven çıkma gibi temel hareketlerde zorluk çekenlerde daha fazla diyabete sahip olma oranı gözlemlenmiştir
plt.rcParams['figure.figsize'] = (10, 5)
df.groupby(['DIFFWALK', 'DIABETES'])['DIABETES'].count().unstack().plot(kind='bar')
plt.ylabel('Diyabet Frekansı')
plt.xlabel('DIFFWALK Sınıfları')
plt.title('Difficulty Walking or Climb Stairs (DIFFWALK vs Diabetes)')
plt.legend(['Non-Diabetes', 'Diabetes'])
plt.show()
# bu grafikte görüyoruz ki yürüme merdive çıkma gibi hareketlerde zorluk yaşamayanlarda daha az diyabet oranı görülüyor
# zorluk yaşayanlarda ise daha fazla diyabet oranı gözlemleniyor


# STROKE Bazında Diyabet Frekansı
plt.rcParams['figure.figsize'] = (15, 7)
df.groupby(['HEARTDISEASEORATTACK', 'DIABETES'])['DIABETES'].count().unstack().plot(kind='bar')
plt.ylabel('Diyabet Frekansı')
plt.xlabel('İnme Durumu Frekansı')
plt.title('İnme Durumuna Göre Diyabet')
plt.legend(['Non-Diabetes', 'Diabetes'])
plt.show()
# inme durumunun diyabeti etkilediğini görmekteyiz


# HIGHBP Bazında Diyabet Frekansı
plt.rcParams['figure.figsize'] = (15, 7)
df.groupby(['HIGHBP', 'DIABETES'])['DIABETES'].count().unstack().plot(kind='bar')
plt.ylabel('Diyabet Frekansı')
plt.xlabel('Yüksek Kolesterol Durumu')
plt.title('Yüksek Kolesterol Durumuna Göre Diyabet')
plt.legend(['Non-Diabetes', 'Diabetes'])
plt.show()
# yüksek kolesterole sahip olanlarda diyabet oranı oldukça fazla gözlemlenmiştir


# yeni değişkenler türetme-feature engineering
################################################################

df['NEW_MENT_GEN'] = df['MENTHLTH'] * df['GENHLTH']
df['NEW_MENT_PHY'] = df['MENTHLTH'] * df['PHYSHLTH']
df['NEW_MENT_DIFF'] = df['MENTHLTH'] * df['DIFFWALK']
df['NEW_PHY_GEN'] = df['PHYSHLTH'] * df['GENHLTH']
df['NEW_PHY_DIFF'] = df['PHYSHLTH'] * df['DIFFWALK']


df['NEW_BMI_HIGHBP'] = df['BMI'] * df['HIGHBP']
df['NEW_BMI_GENHLTH'] = df['BMI'] * df['GENHLTH']
df['NEW_BMI_DIFF'] = df['BMI'] * df['DIFFWALK']


df['NEW_AGE_HEARTDIS'] = df['AGE'] * df['HEARTDISEASEORATTACK']
df['NEW_AGE_HIGHCHOL'] = df['AGE'] * df['HIGHCHOL']
df['NEW_AGE_HIGHBP'] = df['AGE'] * df['HIGHBP']


df['NEW_AGE_DIFF'] = df['AGE'] + df['DIFFWALK']


df['NEW_HEARTDIS_AGE'] = df['HEARTDISEASEORATTACK'] + df['AGE']
df['NEW_HEARTDIS_GEN'] = df['HEARTDISEASEORATTACK'] + df['GENHLTH']
df['NEW_HEARTDIS_DIFF'] = df['HEARTDISEASEORATTACK'] + df['DIFFWALK']
df['NEW_HEARTDIS_STROKE'] = df['HEARTDISEASEORATTACK'] + df['STROKE']
df['NEW_HEARTDIS_HIGHBP'] = df['HEARTDISEASEORATTACK'] + df['HIGHBP']


df['NEW_HIGHBP_DIFF'] = df['HIGHBP'] + df['DIFFWALK']
df['NEW_HIGHBP_HIGHCHOL'] = df['HIGHBP'] + df['HIGHCHOL']
df['NEW_HIGHBP_GENHLTH'] = df['HIGHBP'] + df['GENHLTH']


df['NEW_STROKE_GEN'] = df['STROKE'] + df['GENHLTH']
df['NEW_GEN_DIFF'] = df['GENHLTH'] + df['DIFFWALK']
df['NEW_FRUITS_VEGGIES'] = df['FRUITS'] + df['VEGGIES']
df['NEW_AGE_HEARTDIS'] = df['AGE'] + df['HEARTDISEASEORATTACK']


df['NEW_DIFF_MENTHLTH'] = df['DIFFWALK'] + df['MENTHLTH']
df['NEW_DIFF_AGE'] = df['DIFFWALK'] + df['AGE']


df['NEW_BMI_AGE'] = df['BMI'] * df['AGE']
df['NEW_BMI_HIGHCHOL'] = df['BMI'] + df['HIGHCHOL']

df['NEW_HIGHBP_STROKE'] = df['HIGHBP'] + df['STROKE']
df['NEW_MENTHLTH_AGE'] = df['MENTHLTH'] + df['AGE']
df['NEW_SMOKER_HEARTDIS'] = df['SMOKER'] + df['HEARTDISEASEORATTACK']
df['NEW_SEX_HEARTDIS'] = df['SEX'] + df['HEARTDISEASEORATTACK']


df['NEW_HEALTH'] = df['GENHLTH'] + df['MENTHLTH'] + df['PHYSHLTH']
df['NEW_HEALTH_STROKE'] = df['NEW_HEALTH'] + df['STROKE']
df['NEW_HEALTH_DIFFWALK'] = df['NEW_HEALTH'] + df['DIFFWALK']


df['NEW_PHYSACT_HEARTATTACK'] = df['PHYSACTIVITY'] + df['HEARTDISEASEORATTACK']
df['NEW_PHYSACT_HIGHCHOL'] = df['PHYSACTIVITY'] + df['HIGHCHOL']
df['NEW_PHYSACT_STROKE'] = df['PHYSACTIVITY'] + df['STROKE']


# df te columns isimleri irili ufaklı kolaylık olsun açısından büyütüyoruz
df.columns = [col.upper() for col in df.columns]


df.head()
df.shape  # (70692, 55)


# şimdi grab col names i yeniden çağıracağız çünkü yeni değişkenler oluşturduk
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=15, car_th=25)


# korelasyon matrisi, sayısal değişkenlerin birbirleriyle olan korelasyonu
def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=1, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


correlation_matrix(df, num_cols)


# korelasyon matrisi, hem sayısal hem kategorik değişkenlerin birbirleriyle olan korelasyonu
def correlation_matrix(df):
    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig = sns.heatmap(df.corr(), annot=True, linewidths=1, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


correlation_matrix(df)


# bağımsız sayısal değişkenlerin bağımlı değişkenle olan korelasyonu
std_corr = df.corr()
print(std_corr['DIABETES'].sort_values(ascending=False))


df.head()


# şimdi cat_cols lardan hedef değişkeni çıkarıyoruz
cat_cols = [col for col in cat_cols if "DIABETES" not in col]

df.info()
# label encoding / binary encoding işlemini 2 sınıflı kategorik değişkenlere uyguluyoruz
# yani nominal sınıflı kategorik değişkenlere böylelikle bu iki sınıfı 1-0 şeklinde encodelamış oluyoruz
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes == "float" and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)

# one-hot encoder ise ordinal sınıflı kategorik değişkenler için uyguluyoruz. sınıfları arasında fark olan
# değişkenleri sınıf sayısınca numaralandırıp kategorik değişken olarak df e gönderiyor
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)


df.head(10)


# df_ te columns isimleri irili ufaklı kolaylık olsun açısından büyütüyoruz
df.columns = [col.upper() for col in df.columns]


# Son güncel değişken türlerimi tutuyorum.
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=15, car_th=25)


# Standartlaştırma
#################################################################################
# StandardScaler
X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

# RobustScaler ( Bu Projede Robust Scaler Kullandım)
Robust_df = RobustScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(Robust_df, columns=df[num_cols].columns)

# MinnMaxScaler
mms = MinMaxScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(mms, columns=df[num_cols].columns)
####################################################################################
# RobustScaler
# X_scaled = RobustScaler().fit_transform(X)
# X = pd.DataFrame(X_scaled)


df.head(10)
df.shape


# şimdi veri seti içinden bağımlı ve bağımsız değişkenleri çekiyoruz
y = df["DIABETES"]
X = df.drop(["DIABETES"], axis=1)


check_df(X)
# bu aşamada check_df ile sürekli verimizi kontrol etmemiz gerekiyor çünkü bir problemle karşılaşabiliriz
# bu aşamalardan sonra df te yeni Na değerleri türeyebiliyor dikkatli olmak lazım


# aykırı değer var mı tekrar kontrol ediyoruz, çünkü bazen yukardaki işlemlerden sonra türeyebiliyor
for col in num_cols:
    print(col, check_outlier(df, col))

# aykırı değerleri eşik değerlerle değiştirmek
for col in num_cols:
    replace_with_thresholds(df, col)

# aykırı değer kalmadı
for col in num_cols:
    print(col, check_outlier(df, col))


# modelimizi kuruyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)


def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("CART", DecisionTreeClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier()),
                   ('LightGBM', LGBMClassifier()),
                   ("RF", RandomForestClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=10, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

# ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),


base_models(X, y, scoring="accuracy")
# Base Models....
# accuracy: 0.751 (LR)
# accuracy: 0.7083 (KNN)
# accuracy: 0.6579 (CART)
# accuracy: 0.7487 (Adaboost)
# accuracy: 0.7496 (GBM)
# accuracy: 0.7458 (XGBoost)
# accuracy: 0.75 (LightGBM)
# accuracy: 0.718 (RF)
base_models(X, y, scoring="precision")
# Base Models....
# precision: 0.731 (LR)
# precision: 0.7008 (KNN)
# precision: 0.6667 (CART)
# precision: 0.7355 (Adaboost)
# precision: 0.7334 (GBM)
# precision: 0.7317 (XGBoost)
# precision: 0.74 (LightGBM)
# precision: 0.7032 (RF)
base_models(X, y, scoring="recall")
# Base Models....
# recall: 0.7946 (LR)
# recall: 0.7271 (KNN)
# recall: 0.6376 (CART)
# recall: 0.7766 (Adaboost)
# recall: 0.7846 (GBM)
# recall: 0.786 (XGBoost)
# recall: 0.794 (LightGBM)
# recall: 0.7479 (RF)
base_models(X, y, scoring="f1")
# Base Models....
# f1: 0.7614 (LR)
# f1: 0.7137 (KNN)
# f1: 0.6503 (CART)
# f1: 0.7554 (Adaboost)
# f1: 0.758 (GBM)
# f1: 0.7556 (XGBoost)
# f1: 0.7605 (LightGBM)
# f1: 0.727 (RF)
base_models(X, y, scoring="roc_auc")
# Base Models....
# roc_auc: 0.8269 (LR)
# roc_auc: 0.7691 (KNN)
# roc_auc: 0.6614 (CART)
# roc_auc: 0.8243 (Adaboost)
# roc_auc: 0.8273 (GBM)
# roc_auc: 0.8218 (XGBoost)
# roc_auc: 0.8278 (LightGBM)
# roc_auc: 0.7858 (RF)
######################################################################


##################################################################
logreg = LogisticRegression().fit(X, y)
logreg.get_params()

y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))

#               precision    recall  f1-score   support
#          0.0       0.78      0.71      0.74      7198
#          1.0       0.72      0.80      0.76      6941
#     accuracy                           0.75     14139
#    macro avg       0.75      0.75      0.75     14139
# weighted avg       0.75      0.75      0.75     14139
##################################################################


##################################################################
lgbm = LGBMClassifier().fit(X, y)
lgbm.get_params()
lgbm_params = {"learning_rate": [0.01, 0.1, 0.5],
               "n_estimators": [500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_gs_best = GridSearchCV(lgbm,
                            lgbm_params,
                            cv=10,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

lgbm_gs_best.best_params_
# {'colsample_bytree': 1, 'learning_rate': 0.01, 'n_estimators': 1000}

lgbm = LGBMClassifier(**lgbm_gs_best.best_params_, random_state=17).fit(X, y)  # ya da fit(X_train, y_train)


print(LGBMClassifier(random_state=17).get_params())
y_pred = lgbm.predict(X_test)
print(classification_report(y_test, y_pred))

#               precision    recall  f1-score   support
#            0       0.80      0.72      0.76      7198
#            1       0.73      0.81      0.77      6941

#     accuracy                           0.76     14139
#    macro avg       0.77      0.76      0.76     14139
# weighted avg       0.77      0.76      0.76     14139

y_prob = lgbm.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc_score(y_test, y_pred)
# 0.7639091606830708

# LGBMClassifier ROC Curve
plt.figure(figsize=(5, 5))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='LGBM Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Light GBM Classifier ROC curve')
plt.show()


lightgbm_final = lgbm.set_params(**lgbm_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(lightgbm_final, X, y, cv=3, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()  # 0.7525889079217725
cv_results["test_precision"].mean()  # 0.7318144193043318
cv_results["test_recall"].mean()  # 0.7974027691393928
cv_results["test_f1"].mean()  # 0.7631557789928272
cv_results["test_roc_auc"].mean()  # 0.8290468021999124

# Feature Importance Graphs
features = X.columns
importances = lgbm.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(12, 9))

plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='red', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

##################################################################


# Confusion Matrix Grafiği
##################################################################
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


plot_confusion_matrix(y_test, y_pred)


##################################################################
def plot_confusion_matrix(y, y_pred):
    acc = round(roc_auc_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


plot_confusion_matrix(y_test, y_pred)


#######################################################################
def plot_confusion_matrix(y, y_pred):
    acc = round(f1_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


plot_confusion_matrix(y_test, y_pred)
####################################################################

# Hyperparameter optimization
#####################################################

knn_params = {"n_neighbors": range(2, 50)}
# KNN best params: {'n_neighbors': 45}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}
# CART best params: {'max_depth': 9, 'min_samples_split': 29}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [100, 300, 1000]}
# RF best params: {'max_depth': 15, 'max_features': 'auto', 'min_samples_split': 20, 'n_estimators': 300}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200, 500],
                  "colsample_bytree": [0.5, 0.7, 1]}
# XGBoost best params: {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}

lightgbm_params = {"learning_rate": [0.01, 0.1, 0.5],
               "n_estimators": [100, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}
# LightGBM best params: {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'n_estimators': 500}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]


# Bu fonksiyonda şunu yapıyoruz; öncesindeki hataya bak hiperparametre değerlerini bul,
# hiperparametre değerleri ile sonrasındaki hatasını bul şeklindedir. Çıktılarımız aşağıda buradan gözlemleyelim
def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


best_models = hyperparameter_optimization(X, y)
# Hyperparameter Optimization....
# ########## KNN ##########
# roc_auc (Before): 0.7634
# roc_auc (After): 0.8065
# KNN best params: {'n_neighbors': 49}
# ########## CART ##########
# roc_auc (Before): 0.6643
# roc_auc (After): 0.8176
# CART best params: {'max_depth': 7, 'min_samples_split': 2}
# ########## RF ##########
# roc_auc (Before): 0.7869
# roc_auc (After): 0.826
# RF best params: {'max_depth': 8, 'max_features': 'auto', 'min_samples_split': 15, 'n_estimators': 300}
# ########## XGBoost ##########
# roc_auc (Before): 0.8225
# roc_auc (After): 0.8289
# XGBoost best params: {'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}
# ########## LightGBM ##########
# roc_auc (Before): 0.8286
# roc_auc (After): 0.8295
# LightGBM best params: {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'n_estimators': 1000}
#################################################################################################

best_models = hyperparameter_optimization(X, y, cv=3, scoring="accuracy")
# Hyperparameter Optimization....
# ########## KNN ##########
# accuracy (Before): 0.7048
# accuracy (After): 0.7342
# KNN best params: {'n_neighbors': 37}
# ########## CART ##########
# accuracy (Before): 0.662
# accuracy (After): 0.7415
# CART best params: {'max_depth': 7, 'min_samples_split': 3}
# ########## RF ##########
# accuracy (Before): 0.7196
#####################################################################

best_models = hyperparameter_optimization(X, y, cv=3, scoring="precision")
##############################################################################

best_models = hyperparameter_optimization(X, y, cv=3, scoring="recall")
###########################################################################

best_models = hyperparameter_optimization(X, y, cv=3, scoring="f1")
# Hyperparameter Optimization....
# ########## KNN ##########
# f1 (Before): 0.7451
# f1 (After): 0.7451
# KNN best params: {'n_neighbors': 41}
# ########## CART ##########
# f1 (Before): 0.7498
# f1 (After): 0.7499
# CART best params: {'max_depth': 7, 'min_samples_split': 6}
# ########## RF ##########
# f1 (Before): 0.7577
# f1 (After): 0.7569
# RF best params: {'max_depth': 15, 'max_features': 7, 'min_samples_split': 15, 'n_estimators': 100}
# ########## XGBoost ##########
# f1 (Before): 0.7615
# f1 (After): 0.7615
# XGBoost best params: {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}
# ########## LightGBM ##########
# f1 (Before): 0.7619
# f1 (After): 0.7619
# LightGBM best params: {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'n_estimators': 1000}
##############################################################################


# Stacking & Ensemble Learning
##########################################

# 3 modelin, 3 metrik açısından cross validation hatasına bakacağız bunları değerlendirip ekrana print edeceğiz
# knn, rf, lightgbm hepiniz bi model kurun bi tahminde bulunun ve bir gözlem birimi geldiğinde hep birlikte tahmin yap
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"Precision: {cv_results['test_precision'].mean()}")
    print(f"Recall: {cv_results['test_recall'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


voting_clf = voting_classifier(best_models, X, y)

# Voting Classifier...
# Accuracy: 0.7486137045210208
# Precision: 0.7272746511868361
# Recall: 0.7955921462117356
# F1Score: 0.7598610862332585
# ROC_AUC: 0.826066386734479
##################################################################################


# Prediction for a New Observation
#########################################
import joblib
X.columns

# random user diyerek bir kullanıcı seçiyoruz
random_user = X.sample(1, random_state=1)

# voting_clf ile bu random user ı tahmin ediyoruz
voting_clf.predict(random_user)

# Çalışma kapandığında bu model uçmasın istiyorsak joblib.dumb ile modelimizi kaydediyoruz, isimlendirmesini yapıyoruz
joblib.dump(voting_clf, "diabetes_final_model.pkl")

# bunun çalışıp çalışmadığını nasıl anlarız, yine joblibi kullanıp yükle deriz bu sefer
new_model = joblib.load("diabetes_final_model.pkl")

# new_model olarak kaydetmiştik yüklediğimizi şimdi new model predict dersek random seçtiği bir kullanıcıyı tahmin eder
new_model.predict(random_user)


# Yeni ürettiğimiz değişkenler anlamlı mı
#################################################################################################

# şimdi kurduğumuz modeli kullanarak modelin görmediği test ile test edeceğiz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(15, 15))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


catboost_model = CatBoostClassifier(random_state=46).fit(X_train, y_train)
plot_importance(catboost_model, X_train)


y_pred = catboost_model.predict(X_test)
accuracy_score(y_pred, y_test)
# 0.7512553928849283
precision_score(y_pred, y_test)
recall_score(y_pred, y_test)
f1_score(y_pred, y_test)
########################################################################################


# Değişkenlerin önem düzeyini belirten feature_importance grafiğine bakınız
###################################################################################
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(15, 15))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)

plot_importance(lgbm, X_train)

y_pred = lgbm.predict(X_test)
accuracy_score(y_pred, y_test)
# 0.750406676568357
precision_score(y_pred, y_test)
recall_score(y_pred, y_test)
f1_score(y_pred, y_test)
######################################################################################

