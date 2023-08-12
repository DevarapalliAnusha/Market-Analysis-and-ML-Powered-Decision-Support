##### REAL ESTATE PREDICTION
import pandas as pd
import json
import pickle
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
data=pd.read_excel(r'C:/Users/sai yashasri/Documents/1 python/bangeluru.xlsx')
d1=data.drop(['area_type','availability','balcony','society'],axis='columns')
d2=d1.dropna()
d2['bhk']=d2['size'].apply(lambda x: x.split(' ')[0])
d2=d2.drop(['size'],axis='columns')
d2['location']=d2['location'].str.upper()
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
d3=d2[~d2['total_sqft'].apply(is_float)]
d5=d2[d2['total_sqft'].apply(is_float)]
def convert_sqrt_to_num(x):
    tokens=x.split('-')
    if len(tokens)==2:
        return((float(tokens[0]))+(float(tokens[1])))/2
    try:
        return float(x)
    except:
        return None
d4=d3['total_sqft'].apply(convert_sqrt_to_num)
d3['total_sqft']=d4
d6=pd.concat([d5,d3])
final=d6.copy()
final['price_per_sqft']=final['price']*100000/final['total_sqft']
final.location=final.location.apply(lambda x: x.strip())
location_stats=final.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_less_than_10=location_stats[location_stats<=10]
final.location=final.location.apply(lambda x: 'other' if x in location_less_than_10 else x )
final=final.dropna()
#print(final.isna().sum())
final['total_sqft']=final['total_sqft'].astype(float)
final['bhk']=final['bhk'].astype(float)
final=pd.DataFrame(final)


#### OUTLIER DETECTION AND REMOVAL
#### asuming the minimum area per room is 300
#print(final[(final.total_sqft)/(final.bhk)<300].head())
#print(final.shape)
kaju=final[~((final.total_sqft)/(final.bhk)<300)]
#print(kaju.shape)
#print(kaju.price_per_sqft.describe())
def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))&(subdf.price_per_sqft<(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
pista=remove_pps_outliers(kaju)
def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location)&(df.bhk==2)]
    bhk3=df[(df.location==location)&(df.bhk==3)]
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,color='red',label='3 BHK',marker="*",s=50)
    plt.xlabel('total square feet area')
    plt.ylabel('price per square feet')
    plt.title(location)
    plt.legend()
    #plt.show()
#print(plot_scatter_chart(pista,'Rajaji Nagar'))
#print(pista)
def remove_bhk_outliers(df):
    k=np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
               'mean' :np.mean(bhk_df.price_per_sqft),
               'std'  :np.std(bhk_df.price_per_sqft),
               'count':bhk_df.shape[0]}
        for bhk,bhk_df in location_df.groupby('bhk'):
            stats=bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                k=np.append(k,bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(k,axis='index')
badam=remove_bhk_outliers(pista)
#print(badam.shape)
print(plot_scatter_chart(badam,'Rajaji Nagar'))
plt.hist(badam.price_per_sqft,rwidth=0.5)
plt.xlabel('price per square feet')
plt.ylabel('count')
#plt.show()
#print(badam.bath.unique())
#print(badam[(badam.bath)>(badam.bhk+2)])
raisin=badam[~((badam.bath)>(badam.bhk+2))]
#print(raisin.shape)
plt.hist(raisin.bath,rwidth=0.8)
plt.xlabel('no.of bathrooms')
plt.ylabel('count')
#plt.show()
dummies=pd.get_dummies(raisin.location)
cashew=pd.concat([raisin,dummies.drop('other',axis='columns')],axis='columns')
#print(cashew.head())
mat=cashew.drop(['location'],axis='columns')
cat=mat.drop(['price','price_per_sqft'],axis='columns')
x=pd.DataFrame(cat)
y=pd.DataFrame(mat.price)
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
cvlr=ShuffleSplit(n_splits=5,test_size=0.2,random_state=20)
model_params={
'linear_regression':{'model':LinearRegression(),'params':{'n_jobs':[1,2,3]}},
'lasso':{'model':Lasso(),'params':{'alpha':[1,2],'selection':['random','cycllic']}},
'decision_tree':{'model':DecisionTreeRegressor(),'params':{'criterion':['mse','friedman_mse'],'splitter':['best','random']}}}
scores=[]
#print(x.head())
for model_name,mp in model_params.items():
    dog=GridSearchCV(mp['model'],mp['params'],cv=cvlr,return_train_score=False)
    dog.fit(x,y)
    scores.append({'model' : model_name,'best_score' : dog.best_score_,'best_params' : dog.best_params_ })
pat=pd.DataFrame(scores,columns=['model','best_score','best_params'])
hat=LinearRegression()
hat.fit(x,y)
#print(pat.iloc[0:3,1:3])
pau=(x.columns)
print(pau)
#pau=pat.upper()
def predict_price(location,sqft,bhk,bath):
    kap=np.where(pau==location)[0][0]
    lat=np.zeros(len(pau))
    lat[0]=sqft
    lat[1]=bath
    lat[2]=bhk
    if kap>=0:
        lat[kap]=1
    return hat.predict([lat])
abb=str(input('enter the location : '))
a=abb.upper()
b=(input('enter the area sqft : '))
c=input('enter the bhk : ')
d=input('enter the bath : ')
print(predict_price(a,b,c,d))


joblib.dump(hat,'Real Estate For Banglore City')

columns={'data_columns':[col.lower() for col in x.columns]}
with open('columns.json','w') as f:
    f.write(json.dumps(columns))