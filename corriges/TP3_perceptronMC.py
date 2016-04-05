
import numpy as np
from sklearn.datasets import load_iris


irisData=load_iris()


X=irisData.data
Y=irisData.target

X=X[0:120,:]
Y=Y[0:120]


nc = len(irisData.target_names)


training_set=[]
for i in range(len(X[:,1])):
    training_set.append((tuple(X[i,:]),Y[i]))



d = len(training_set[0][0])
learning_rate = 1

weights = np.zeros((nc,d))


def dot_product(values, weights):
    return sum(value * weight for value, weight in zip(values,weights))


res = np.zeros(nc)
iter=0
while True:
    iter+=1
    print('-' * 60)
    error_count = 0
    for input_vector, desired_output in training_set:
        for c in range(nc):
            res[c] = dot_product(input_vector,weights[c])
        result = np.argmax(res)
        error = desired_output - result
        if error != 0:
            error_count +=1
            for index, value in enumerate(input_vector):
                weights[result][index] += -learning_rate * value
                weights[desired_output][index] += learning_rate * value
    if error_count == 0:
        break
    print(iter)

print(iter)
print(weights)