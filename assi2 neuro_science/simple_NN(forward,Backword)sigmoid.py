import random

def sigmoid(x):
    return 1 / (1 + 2.718281828459045 ** -x)

def sigmoid_derivative(x):
    return x * (1 - x)

def random_weight():
    return random.uniform(-.5, .5)

def weighted_sum(inputs, weights, bias):
    return sum(i * w for i, w in zip(inputs, weights)) + bias

w1, w2, w3, w4 = random_weight(), random_weight(), random_weight(), random_weight()
w5, w6, w7, w8 = random_weight(), random_weight(), random_weight(), random_weight()

i1, i2 = .05, .1
b1, b2 = .5, .7
learning_rate = 0.5  

# Hidden layer 
net_h1 = weighted_sum([i1, i2], [w1, w2], b1) 
out_h1 = sigmoid(net_h1)
net_h2 = weighted_sum([i1, i2], [w3, w4], b1) 
out_h2 = sigmoid(net_h2)

# Output layer 
net_o1 = weighted_sum([out_h1, out_h2], [w5, w6], b2) 
out_o1 = sigmoid(net_o1)
net_o2 = weighted_sum([out_h1, out_h2], [w7, w8], b2) 
out_o2 = sigmoid(net_o2)

print("Forward phase ")
print(f"prob of classs o1 : {out_o1}")
print(f"prob of classs o2 : {out_o2}")
print("It is class o1") if (out_o1 > out_o2) else print("It is class o2")
print("\n\nWeights:")
print(f"W1: {w1}, W2: {w2}, W3: {w3}, W4: {w4}")
print(f"W5: {w5}, W6: {w6}, W7: {w7}, W8: {w8}")

actual_o1, actual_o2 = 0.01, 0.99 

error_o1 = .5 * (out_o1 - actual_o1) ** 2
error_o2 = .5 * (out_o2 - actual_o2) ** 2
total_error = error_o1 + error_o2

delta_o1 = -(actual_o1 - out_o1) * sigmoid_derivative(out_o1)
delta_o2 = -(actual_o2 - out_o2) * sigmoid_derivative(out_o2)

delta_h1 = (delta_o1 * w5 + delta_o2 * w7) * sigmoid_derivative(out_h1)
delta_h2 = (delta_o1 * w6 + delta_o2 * w8) * sigmoid_derivative(out_h2)

w1 += learning_rate * delta_h1 * i1
w2 += learning_rate * delta_h1 * i2
w3 += learning_rate * delta_h2 * i1
w4 += learning_rate * delta_h2 * i2

w5 += learning_rate * delta_o1 * out_h1
w6 += learning_rate * delta_o1 * out_h2
w7 += learning_rate * delta_o2 * out_h1
w8 += learning_rate * delta_o2 * out_h2

print('\n\nBackpropagation')
print("Updated Weights:")
print(f"W1: {w1}, W2: {w2}, W3: {w3}, W4: {w4}")
print(f"W5: {w5}, W6: {w6}, W7: {w7}, W8: {w8}")
