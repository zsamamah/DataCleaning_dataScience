from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#splitng the data into training and testing set
df = pd.read_csv("Car_Acc_Students.csv")

#remove "Unnamed 0" and "Year" columns
df.drop("Unnamed: 0", axis=1, inplace=True)
df.drop("Year", axis=1, inplace=True)

#drop_duplicates
df.drop_duplicates(inplace=True)

# Replace null values with median if exists
cols = df.columns
for col in cols:
    if df[col].isnull().sum()!= 0:
        print(col)
        print("Before: ",df[col].isnull().sum())
        df[col].fillna(df[col].mean(), inplace=True)
        print("After: ",df[col].isnull().sum())
        
# Detecting and filtering outliers
#Example (example in "Simple Injuries "42" is outlier), replacing it another normal maxiumum value in that column
z_scores = (df['Simple Injuries'] - df['Simple Injuries'].mean()) / df['Simple Injuries'].std()
threshold = 3
outliers = df[abs(z_scores) > threshold]
max_value = df[abs(z_scores) <= threshold]['Simple Injuries'].max()
df.loc[abs(z_scores) > threshold, 'Simple Injuries'] = max_value
#train_set['Simple Injuries'].replace(42,10,inplace=True)

#replace the '0' value in 'Drive Age' with median in training set

median_driver_age = df['Driver Age'].median()
df['Driver Age'] = df['Driver Age'].replace(0, median_driver_age)

# check and filter the maximum value in "driver age"
z_scores = (df['Driver Age'] - df['Driver Age'].mean()) / df['Driver Age'].std()
threshold = 3
outliers = df[abs(z_scores) > threshold]
max_value = df[abs(z_scores) <= threshold]['Driver Age'].max()
df.loc[abs(z_scores) > threshold, 'Driver Age'] = max_value

#Handling text attributes using One Hot Encoding 
#for the following features : "Accident Type" , "Driver Licnesee Type", "City", "Driver Sex", "Car"

df['Accident Type'] = OneHotEncoder().fit_transform(df[['Accident Type']]).toarray()
df['Driver Licnesee Type'] = OneHotEncoder().fit_transform(df[['Driver Licnesee Type']]).toarray()
df['City'] = OneHotEncoder().fit_transform(df[['City']]).toarray()
df['Driver Sex'] = OneHotEncoder().fit_transform(df[['Driver Sex']]).toarray()
df['Car'] = OneHotEncoder().fit_transform(df[['Car']]).toarray()

df['Road Surfuce Description'] = OneHotEncoder().fit_transform(df[['Road Surfuce Description']]).toarray()
df['Weather'] = OneHotEncoder().fit_transform(df[['Weather']]).toarray()

#Handling text attributes using Ordinal Encoding Encoding for other features:

#Ordinal encoding for for the 'Road Lanes' column, this is a manual ordinal encoding
road_lanes_mapping = {'Parking lot':4, 'Public Parks':5, 'One way':1, 'Two ways':2, 'Two ways with island':3}
df['Road Lanes'] = df['Road Lanes'].map(road_lanes_mapping)

#for "Light"
Light_mapping = {'day':2, 'lighty night':4, 'dark night':6, 'sunset':3, 'sunrise':1, 'dark':5}
df['Light'] = df['Light'].map(Light_mapping)

#print data frame
print(df)