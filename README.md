# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
###  Developed by: PUGAZH V
### RegisterNumber: 212225240109
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Required Libraries**: Bring in essential libraries such as pandas, numpy, matplotlib, and sklearn.  
2. **Dataset Loading**: Load the dataset containing car prices and associated features.  
3. **Data Preparation**: Address missing data and perform feature selection if needed.  
4. **Data Splitting**: Divide the dataset into training and testing subsets.  
5. **Model Training**: Develop a linear regression model and train it using the training data.  
6. **Prediction Generation**: Apply the model to predict outcomes for the test dataset.  
7. **Model Evaluation**: Evaluate the model's performance using metrics like R² score, Mean Absolute Error (MAE), etc.  
8. **Assumption Verification**: Analyze residual plots to check for homoscedasticity, normal distribution, and linearity.  
9. **Result Presentation**: Display the predictions and performance evaluation metrics.  

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#Load the dataset
df = pd.read_csv('CarPrice_Assignment (1).csv')
df.head()
#select features and target
X = df[['enginesize', 'horsepower', 'citympg', 'highwaympg']] 
y = df['price']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

#Predictions
y_pred = model.predict(X_test_scaled)
#Model coefficients and metrics
#print("="*50)
print('Name: PUGAZH V')
print('Reg No: 2122252400109')
print("MODEL COEFFIECIENTS:")
for feature, coef in zip (X.columns, model.coef_):
    print(f"{feature:>12}: {coef:10.2f}")
print(f"{'Intercept':>12}: {model.intercept_:>10.2f}")

print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(y_test, y_pred):>10.2f}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(y_test, y_pred)):>10.2f}")
print(f"{'R-squared':>12}: {r2_score(y_test, y_pred):>10.2f}")
print(f"{'MAE':>12}: {mean_absolute_error(y_test, y_pred):>10.2f}")
#print("="*50)


#1. lINEARITY CHECK
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

#2. Independence (Durbin-Watson)
residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}",
      "\n(Values close to 2 indicate no autocorrelation)")

#3. Homoscedasticity
plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title ("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

#4. Normality of residuals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```

## Output:
<img width="951" height="693" alt="Screenshot 2026-02-04 092912" src="https://github.com/user-attachments/assets/6ae6fbe0-75d1-4eb5-bd8b-f98f50f77f7c" />
<img width="944" height="505" alt="Screenshot 2026-02-04 092925" src="https://github.com/user-attachments/assets/35a75845-45ce-4c0c-9b65-6ab24cba8843" />
<img width="1003" height="406" alt="Screenshot 2026-02-04 092937" src="https://github.com/user-attachments/assets/1ed7d009-d433-49aa-ad40-c4d5603801dd" />




## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
