#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[2]:


file_path = "D:\DSML\individual+household+electric+power+consumption\household_power_consumption.txt"
df = pd.read_csv(file_path)
print(df.head())


# In[4]:


df = pd.read_csv(file_path, sep=';', parse_dates={'datetime': ['Date', 'Time']}, infer_datetime_format=True, low_memory=False, na_values=['nan','?'])
print(df.head)


# In[5]:


print(df.info())


# In[6]:


print(df.describe())


# In[9]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.boxplot(df['Global_intensity'].dropna())
plt.title('Boxplot of Global Intensity')
plt.ylabel('Global Intensity')
plt.show()


# In[10]:


plt.figure(figsize=(10, 6))
plt.scatter(df['Global_active_power'], df['Global_intensity'], alpha=0.5)
plt.title('Global Active Power vs Global Intensity')
plt.xlabel('Global Active Power (kilowatts)')
plt.ylabel('Global Intensity (amperes)')
plt.show()


# In[11]:


df.dropna(inplace=True)


# In[12]:


duplicates = df[df.duplicated()]
print(f'Number of duplicate rows: {duplicates.shape[0]}')


# In[15]:


import seaborn as sns
plt.figure(figsize=(10, 6))
sns.histplot(df['Global_intensity'], bins=50, kde=True)
plt.title('Distribution of Global Intensity')
plt.xlabel('Global Intensity')
plt.ylabel('Frequency')
plt.show()


# In[16]:


plt.figure(figsize=(10, 6))
plt.scatter(df['Global_active_power'], df['Global_intensity'], alpha=0.5)
plt.title('Scatter Plot of Global Active Power vs. Global Intensity')
plt.xlabel('Global Active Power')
plt.ylabel('Global Intensity')
plt.show()


# In[19]:


df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month


# In[20]:


from sklearn.model_selection import train_test_split
X = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'hour', 'day_of_week', 'month']]
y = df['Global_intensity']


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[23]:


model = LinearRegression()
model.fit(X_train_scaled, y_train)


# In[ ]:


y_pred = model.predict(X_test_scaled)


# In[ ]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'R2 Score: {r2}')


# In[ ]:


from sklearn.svm import SVR
svr_model = SVR(kernel='linear')
svr_model.fit(X_train_scaled,y_train)


# In[ ]:


mae_svr = mean_absolute_error(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)
print("SVR Model Evaluation:")
print(f'MAE: {mae_svr}')
print(f'MSE: {mse_svr}')
print(f'R2 Score: {r2_svr}')


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


# In[ ]:


mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("RandomForestRegressor Model Evaluation:")
print(f'MAE: {mae_rf}')
print(f'MSE: {mse_rf}')
print(f'R2 Score: {r2_rf}')


# In[ ]:




