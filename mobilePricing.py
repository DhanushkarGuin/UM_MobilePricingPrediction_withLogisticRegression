# Mobile Phone Price Prediction using Linear Regression
# 0 = Low, 1 = Medium, 2 = High, 3 = Very High

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train , X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)

Y_pred = reg.predict(X_test)
df = pd.DataFrame({'Real Values': Y_test, 'Predicted Values': Y_pred})
print(df)

from sklearn import metrics 
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))

battery_power = int(input("Enter power of your battery:"))
bluetooth = int(input("Does your mobile have bluetooth(Yes=1 and No=0):"))
clock_speed = float(input("Enter processor speed:"))
dual_sim = int(input("Does your mobile have dual sim slot(Yes=1 and No=0):"))
front_camera = int(input("Megapixels of your front camera:"))
fourG = int(input("Does your mobile support 4g(Yes=1 and No=0):"))
internal_memory = int(input("Enter size of memory in GB:"))
mobile_depth = float(input("Enter the depth of your mobile in cms:"))
mobile_weight = int(input("Enter the weight of your mobile in gms:"))
number_of_cores = int(input("Enter the number of cores:"))
primary_camera = int(input("Megapixels of your primary camera:"))
resolution_ht = int(input("Height of resolution:"))
resolution_wt = int(input("Width of resolution:"))
ram = int(input("Rams in MB:"))
screen_height = int(input("Enter screen height in cms:"))
screen_width = int(input("Enter screen width in cms:"))
screen_time = int(input("Enter screen time(in a single charge):"))
threeG = int(input("Does your mobile support 3g(Yes=1 and No=0):"))
touch_screen = int(input("Is your mobile touch screen(Yes=1 and No=0):"))
wifi= int(input("Does your mobile support wifi(Yes=1 and No=0):"))

print("\nYour mobile's price range will be on a scale of 0-3 where 0 is lowest and 3 is highest!\n")

custom_input = pd.DataFrame([[battery_power,bluetooth,clock_speed,dual_sim,front_camera,fourG,internal_memory,mobile_depth,mobile_weight,number_of_cores,
                            primary_camera,resolution_ht,resolution_wt,ram,screen_height,screen_width,screen_time,threeG,touch_screen,wifi]],
                            columns=X.columns)
predicted_price = reg.predict(custom_input)
print(f"Predicted mobile phone price: {predicted_price[0]:.2f}")

