import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv(' data.csv')
df['G'] = df['GDP']/df['CPI']
df['Y'] = df['Invest']/df['CPI']
df['P'] = (df['CPI'] - df['CPI'].shift(1)) / df['CPI'].shift(1)
df['t'] = df['year'] - 1990
df = df.drop(0)
df = df.reset_index(drop=True)
print(df.head())
X = df[['t', 'G', 'R', 'P']]
y = df['Y']
# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LinearRegression()
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)
r_squared = r2_score(y, y_pred)
print("R-squared:", r_squared)
print("回归系数:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")
print("截距:", model.intercept_)
