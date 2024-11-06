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
![Screenshot 2024-11-05 113358](https://github.com/user-attachments/assets/27101741-3fc2-473f-a18f-91c8ff648882)
~~~
df.dropna()
~~~
![Screenshot 2024-11-05 113433](https://github.com/user-attachments/assets/ca3b128f-8b08-419f-96d1-164952a269fe)
~~~
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
~~~
![Screenshot 2024-11-05 113516](https://github.com/user-attachments/assets/006cfd37-0e26-4407-9fe4-1557518b6bc6)
~~~
df1=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height', 'Weight']]=sc.fit_transform(df1[['Height', 'Weight']])
df1.head(10)
~~~
![Screenshot 2024-11-05 113524](https://github.com/user-attachments/assets/ac450b53-1e0e-480f-86bc-349e5260da4b)
~~~
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height', 'Weight']]=scaler.fit_transform(df[['Height', 'Weight']])
df.head(10)
~~~
![Screenshot 2024-11-05 113541](https://github.com/user-attachments/assets/5e87d539-b34d-4f6e-95ee-63cb4ad75ece)
~~~
df2=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height', 'Weight']]=scaler.fit_transform(df2[['Height', 'Weight']])
df2
~~~
![Screenshot 2024-11-05 113617](https://github.com/user-attachments/assets/2556e66f-3c0c-4435-8771-e7598fa70205)
~~~
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height', 'Weight']]=scaler.fit_transform(df3[['Height', 'Weight']])
df3
~~~
![Screenshot 2024-11-05 113632](https://github.com/user-attachments/assets/26a54fba-d2b2-4d21-9cfc-5fd503386abe)
~~~
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height', 'Weight']]=scaler.fit_transform(df4[['Height', 'Weight']])
df4.head()
~~~
![Screenshot 2024-11-05 113641](https://github.com/user-attachments/assets/2144bf69-a10d-40da-8dbd-cd3f81b28398)
~~~
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv", na_values=[" ?"])
data
~~~
![Screenshot 2024-11-05 113657](https://github.com/user-attachments/assets/4336230a-caae-46e5-bc2a-7c635dc040bb)
~~~
data.isnull().sum()
missing=data[data.isnull().any(axis=1)]
missing
~~~
![Screenshot 2024-11-05 113747](https://github.com/user-attachments/assets/4475f19a-52f4-40b1-9802-f985bc4eac3a)
~~~
data2=data.dropna(axis=0)
data2
~~~
![Screenshot 2024-11-05 113756](https://github.com/user-attachments/assets/848ec85f-ce03-41b6-b8d6-7d1167e74d1f)
~~~
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000' : 0, ' greater than 50,000' : 1})
print(data2['SalStat'])
~~~
![Screenshot 2024-11-05 113831](https://github.com/user-attachments/assets/ce77aaaa-8abe-4321-894c-2d35ff9d3eef)
~~~
sal2=data2['SalStat']
dfs=pd.concat([sal , sal2], axis=1)
dfs
~~~
![Screenshot 2024-11-05 113837](https://github.com/user-attachments/assets/bd6754ea-33c6-4616-a83d-8ebe7b9b16ea)
~~~
data2
~~~
![Screenshot 2024-11-05 113850](https://github.com/user-attachments/assets/6cf1a855-9bdf-432c-a821-ff5c9f971789)
~~~
new_data=pd.get_dummies(data2, drop_first=True)
new_data
~~~
![Screenshot 2024-11-05 113850](https://github.com/user-attachments/assets/6cf1a855-9bdf-432c-a821-ff5c9f971789)
~~~
columns_list=list(new_data.columns)
print(columns_list)
~~~
![Screenshot 2024-11-05 113901](https://github.com/user-attachments/assets/f7a860b8-8d2b-4ee1-97c3-ee198b63f056)
~~~
y=new_data['SalStat'].values
print(y)
~~~
![Screenshot 2024-11-05 113913](https://github.com/user-attachments/assets/f7f0a47f-c38d-44d8-900f-edc55101b12b)
~~~
x=new_data[features].values
print(x)
~~~
![Screenshot 2024-11-05 113947](https://github.com/user-attachments/assets/b9f23b0a-4fe0-4c3c-9988-cda81338aa8e)
~~~
train_x, test_x, train_y, test_y=train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x, train_y)
~~~
![Screenshot 2024-11-05 113952](https://github.com/user-attachments/assets/f101322a-5ac6-4842-a161-6453bf2edeee)
~~~
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
~~~
![Screenshot 2024-11-05 114003](https://github.com/user-attachments/assets/77bbfd46-cf9c-4fe9-ad3c-d4fafff74314)
~~~
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
~~~
![Screenshot 2024-11-05 114007](https://github.com/user-attachments/assets/e9d91d71-2732-46ff-98bc-60eb45b20a4a)
~~~
print('Misclassified samples: %d' % (test_y !=prediction).sum())
~~~
![Screenshot 2024-11-05 114011](https://github.com/user-attachments/assets/1de8d60a-fe00-4db3-9ff3-d4a94044d0c5)
~~~
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
~~~
![Screenshot 2024-11-05 114030](https://github.com/user-attachments/assets/dd4afde2-568a-41bf-ad8c-61ebe93750d8)
~~~
contingency_table=pd.crosstab(tips['sex'], tips['time'])
print(contingency_table)
~~~
![Screenshot 2024-11-05 114058](https://github.com/user-attachments/assets/1da05760-1075-43a3-959b-c678f6d07db8)
~~~
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f'Chi-Square Statistic: {chi2}')
print(f'P-value: {p}')
~~~
![Screenshot 2024-11-05 114210](https://github.com/user-attachments/assets/0e18939a-e987-48ce-9864-4f79dd2e35e9)
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
![Screenshot 2024-11-05 114218](https://github.com/user-attachments/assets/9e65a07b-fda6-42b9-8ddb-53b362007da0)

# RESULT:
Finally,perform Feature Scaling and Feature Selection process is executed successfully.


