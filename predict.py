import pickle
import numpy as np
import pandas as pd

pipeline = pickle.load(open('pipeline.pkl','rb'))

columns = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi']

## Assuming User Input
test_input = pd.DataFrame([[1000, 1, 2.1, 1, 13, 1, 33, 0.6, 139, 8, 2, 1263, 1716, 3220, 16, 3, 19, 1, 1, 1]],columns=columns)

## Prediciton
prediction = pipeline.predict(test_input)
print(prediction)