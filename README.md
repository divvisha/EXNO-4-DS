# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score.
2. The features are then rescaled with x̄ =0 and σ=1
3. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
4. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
5. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
~~~
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
~~~
![image](https://github.com/user-attachments/assets/39c05fab-6d7c-4cbb-8820-94c508d0f5c8)
~~~
df.dropna()
~~~
![image](https://github.com/user-attachments/assets/acd68e50-2a45-4511-b583-ab0f85751f6f)
~~~
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
~~~
![image](https://github.com/user-attachments/assets/b696f04e-baab-40a6-86a1-762b8ca168ab)
~~~
df1=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height', 'Weight']]=sc.fit_transform(df1[['Height', 'Weight']])
df1.head(10)
~~~
![image](https://github.com/user-attachments/assets/2a9d36c0-0a15-4f82-bae1-f47a810332ae)
~~~
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height', 'Weight']]=scaler.fit_transform(df[['Height', 'Weight']])
df.head(10)
~~~
![image](https://github.com/user-attachments/assets/4ddf1c87-8dcc-435f-8ce6-883ff71928e7)
~~~
df2=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height', 'Weight']]=scaler.fit_transform(df2[['Height', 'Weight']])
df2
~~~
![image](https://github.com/user-attachments/assets/c584524e-148d-412b-ba56-d15c2ef058e0)
~~~
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height', 'Weight']]=scaler.fit_transform(df3[['Height', 'Weight']])
df3
~~~
![image](https://github.com/user-attachments/assets/de7fcd30-eec6-4e5b-8637-27bfd5642ea0)
~~~
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height', 'Weight']]=scaler.fit_transform(df4[['Height', 'Weight']])
df4.head()
~~~
![image](https://github.com/user-attachments/assets/522c6470-6bf1-4779-801d-84d335103957)
~~~
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv", na_values=[" ?"])
data
~~~
![image](https://github.com/user-attachments/assets/d4d97e63-a355-4a6b-92b1-d1283f768fe6)
~~~
data.isnull().sum()
~~~
~~~
missing=data[data.isnull().any(axis=1)]
missing
~~~
![image](https://github.com/user-attachments/assets/befbd677-2353-444f-b8b5-b357e419dcbd)
~~~
data2=data.dropna(axis=0)
data2
~~~
![image](https://github.com/user-attachments/assets/8ca4df1e-5d77-42a6-9b08-ecc7f257c068)
~~~
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000' : 0, ' greater than 50,000' : 1})
print(data2['SalStat'])
~~~
![image](https://github.com/user-attachments/assets/7362148c-fa33-4d11-b022-01ae1078b6e1)
~~~
sal2=data2['SalStat']
dfs=pd.concat([sal , sal2], axis=1)
dfs
~~~
![image](https://github.com/user-attachments/assets/1bdcd3e8-a34a-4bfd-94ad-3d2a313dd6a8)
~~~
data2
~~~
![image](https://github.com/user-attachments/assets/d4e4da80-d96e-4f6d-af60-3b0d571c806d)
~~~
new_data=pd.get_dummies(data2, drop_first=True)
new_data
~~~
![image](https://github.com/user-attachments/assets/509c7bd1-a391-42bc-ba66-a8f1a13bd13f)
~~~
columns_list=list(new_data.columns)
print(columns_list)
~~~
![image](https://github.com/user-attachments/assets/d8627461-e051-4f2c-a9d3-8fecfd74cbb3)
~~~
y=new_data['SalStat'].values
print(y)
~~~
![image](https://github.com/user-attachments/assets/1bad9427-65a1-48f2-a9c2-f48f03215cdd)
~~~
x=new_data[features].values
print(x)
~~~
![image](https://github.com/user-attachments/assets/cc8dd6d9-90a4-4f57-a2a9-39f9e215085b)
~~~
train_x, test_x, train_y, test_y=train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x, train_y)
~~~
![image](https://github.com/user-attachments/assets/64d10502-f031-4427-b2a2-7a6f3c7b5752)
~~~
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
~~~
![image](https://github.com/user-attachments/assets/c9d7ec85-154c-4001-8606-623d79d751e0)
~~~
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
~~~
![image](https://github.com/user-attachments/assets/82549214-136f-43d7-a7cc-c878fd400536)
~~~
print('Misclassified samples: %d' % (test_y !=prediction).sum())
~~~
![image](https://github.com/user-attachments/assets/51ae126f-e9a5-4a21-aaa6-a6d6fb5abc03)
~~~
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
~~~
![image](https://github.com/user-attachments/assets/b9e95592-30d3-4352-844d-c41f3e209af7)
~~~
contingency_table=pd.crosstab(tips['sex'], tips['time'])
print(contingency_table)
~~~
![image](https://github.com/user-attachments/assets/ca38d5f8-4ca8-49db-a263-1e5a4b2829a2)
~~~
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f'Chi-Square Statistic: {chi2}')
print(f'P-value: {p}')
~~~
![image](https://github.com/user-attachments/assets/4a746b12-330a-47db-b0bd-75b4d8d98649)
~~~
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': ['A', 'B', 'C', 'A', 'B'],
    'Feature3': [0, 1, 1, 0, 1],
    'Target': [0, 1, 1, 0, 1]
}
df=pd.DataFrame(data)
X=df[['Feature1', 'Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif, k=1)
X_new=selector.fit_transform(X,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print('Selected Features:')
print(selected_features)
~~~
![image](https://github.com/user-attachments/assets/129424f0-0447-43c7-af7c-ad08823e8780)

# RESULT:
Finally,perform Feature Scaling and Feature Selection process is executed successfully.


