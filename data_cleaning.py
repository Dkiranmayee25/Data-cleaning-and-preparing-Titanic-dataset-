import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Titanic-Dataset.csv')
print("\nGiven Dataset:\n", df.head())
print("\nData Types:\n", df.dtypes)
print("\nNull Values:\n", df.isnull().sum())

df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)
df['Age'] = SimpleImputer(strategy='median').fit_transform(df[['Age']])
df['Embarked'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['Embarked']]).ravel()

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['Age'])
plt.title('Boxplot of Age')
plt.subplot(1, 2, 2)
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare')
plt.tight_layout()
plt.show()

def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

df = remove_outliers(df, 'Age')
df = remove_outliers(df, 'Fare')

print("\nFinal dataset shape after cleaning:", df.shape)
df.to_csv('Cleaned_Titanic_Data.csv', index=False)
df.to_excel('Cleaned_Titanic_Data.xlsx', index=False)
print("\nCleaned data saved as:")
print(" - Cleaned_Titanic_Data.csv")
print(" - Cleaned_Titanic_Data.xlsx")
print("\nPreview of Cleaned Data:\n", df.head())