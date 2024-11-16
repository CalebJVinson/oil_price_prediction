### New Analysis Model

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import statsmodels.api as sm

# initialize our constants
path = 'C:/Users/Owner/Downloads/'
file = 'UsedCarData.xlsx'

# read in our dataframe
df_car = pd.read_excel(path + file)

# check datatypes
df_car.dtypes

# check value counts for diff columns
df_car['Transmission'].value_counts()
df_car['Drivetrain'].value_counts()
df_car['Make'].value_counts()
df_car['Model'].value_counts()
df_car['Engine'].value_counts()
df_car['Fuel_Type'].value_counts()
df_car['Body_Type'].value_counts()

### do some munging by mapping and converting to int64
df_car['Engine'][df_car['Engine'] == 'E'] = 0
df_car['Engine'] = df_car['Engine'].astype('int64')

df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Gas/Electric Hybrid'] = 'Gasoline Hybrid'
df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Gasoline Fuel'] = 'Gas'
df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Gaseous Fuel Compatible'] = 'Flexible'
df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Other'] = 'Flexible'

df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Gas'] = 0
df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Gasoline Hybrid'] = 1
df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Electric'] = 2
df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Flexible'] = 3
df_car['Fuel_Type'][df_car['Fuel_Type'] == 'Diesel'] = 4
df_car['Fuel_Type'] = df_car['Fuel_Type'].astype('int64')

df_car['Drivetrain'][df_car['Drivetrain'] == 'AWD'] = 0
df_car['Drivetrain'][df_car['Drivetrain'] == 'FWD'] = 1
df_car['Drivetrain'][df_car['Drivetrain'] == '4x4'] = 2
df_car['Drivetrain'][df_car['Drivetrain'] == 'RWD'] = 3
df_car['Drivetrain'][df_car['Drivetrain'] == '4WD'] = 4
df_car['Drivetrain'][df_car['Drivetrain'] == '2WD'] = 5
df_car['Drivetrain'] = df_car['Drivetrain'].astype('int64')

# correlation matrix
df_corr = df_car.corr()

# Regression 1
model=sm.OLS(df_car.Price, df_car[['Year', 'Kilometres', 'Engine', 'Drivetrain', 'Passengers', 'Doors', 'Fuel_Type', 'City', 'Highway']])
result=model.fit()
result.summary()

variables=result.model.exog
vif=[variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif

# Regression 2
model=sm.OLS(df_car.Price, df_car[['Year', 'Kilometres', 'Engine', 'Passengers', 'Doors', 'Fuel_Type', 'City', 'Highway']])
result=model.fit()
result.summary()

variables=result.model.exog
vif=[variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif

# Regression 3
model=sm.OLS(df_car.Price, df_car[['Year', 'Kilometres', 'Engine', 'Passengers', 'Doors', 'Fuel_Type', 'Highway']])
result=model.fit()
result.summary()

variables=result.model.exog
vif=[variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif

# Regression 4
model=sm.OLS(df_car.Price, df_car[['Year', 'Kilometres', 'Passengers', 'Doors', 'Fuel_Type', 'Highway']])
result=model.fit()
result.summary()

variables=result.model.exog
vif=[variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif

# Regression 5
model=sm.OLS(df_car.Price, df_car[['Kilometres', 'Passengers', 'Doors', 'Fuel_Type', 'Highway']])
result=model.fit()
result.summary()

variables=result.model.exog
vif=[variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif

# Regression 6
model=sm.OLS(df_car.Price, df_car[['Kilometres', 'Passengers', 'Fuel_Type', 'Highway']])
result=model.fit()
result.summary()

variables=result.model.exog
vif=[variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif

# Regression 7
model=sm.OLS(df_car.Price, df_car[['Kilometres', 'Fuel_Type', 'Highway']])
result=model.fit()
result.summary()

variables=result.model.exog
vif=[variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif
