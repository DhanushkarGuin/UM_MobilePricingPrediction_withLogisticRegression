import pickle
import numpy as np

reg = pickle.load(open('reg.pkl','rb'))

## Assuming User Input
test_input = np.array([1000, 1, 2.1, 1, 13, 1, 33, 0.6, 139, 8, 2, 1263, 1716, 3220, 16, 3, 19, 1, 1, 1], dtype=object).reshape(1,20)

## Prediciton
prediction = reg.predict(test_input)
print(prediction)
