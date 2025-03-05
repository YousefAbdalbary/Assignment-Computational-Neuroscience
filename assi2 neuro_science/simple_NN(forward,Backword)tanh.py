import random
def tanh(x):
    exp_plus_x = 2.718281828459045 ** x
    exp_neg_x = 2.718281828459045 ** -x
    return (exp_plus_x - exp_neg_x) / (exp_plus_x + exp_neg_x)
def random_weight():
    return random.uniform(-.5,.5)
def weighted_sum (inputs,weigths, bias):
    return sum(i * w for i,w in zip(inputs,weigths))+bias

def tanh_derivative(x):
    return 1- x**2

w1,w2,w3,w4 = random_weight(),random_weight(),random_weight(),random_weight()
w5,w6,w7,w8 = random_weight(),random_weight(),random_weight(),random_weight()

i1,i2 = .05,.1
b1,b2 =.5 , .7
learning_rate= 0.5  

#hidden layer 
net_h1 = weighted_sum([i1,i2],[w1,w2],b1 ) 
out_h1 = tanh(net_h1)
net_h2 = weighted_sum([i1,i2],[w3,w4],b1 ) 
out_h2 = tanh(net_h2)

#output layer 
net_o1 = weighted_sum([out_h1,out_h2],[w5,w6],b2 ) 
out_o1 = tanh(net_o1)
net_o2 = weighted_sum([out_h1,out_h2],[w7,w8],b2 ) 
out_o2 = tanh(net_o2)

print("Forward phase ")
print(f"prob of classs o1 : {out_o1}")
print(f"prob of classs o2 : {out_o2}")
print("It is class o1") if (out_o1 > out_o2) else print("It is class o2")
print("\n\nWeights:")
print(f"W1: {w1}, W2: {w2}, W3: {w3}, W4: {w4}")
print(f"W5: {w5}, W6: {w6}, W7: {w7}, W8: {w8}")
actual_o1, actual_o2 = 0.01, 0.99 

error_o1 = .5 *(out_o1 - actual_o1)**2
error_o2 = .5 *(out_o2 - actual_o2)**2
total_error =error_o1 + error_o2

delta_o1 = -(actual_o1 - out_o1) * tanh_derivative(net_o1)
delta_o2 = -(actual_o2 - out_o2) * tanh_derivative(net_o2)

deltaH1 = (delta_o1 * w5 + delta_o2 * w7) * tanh_derivative(net_h1)
deltaH2 = (delta_o1 * w6 + delta_o2 * w8) * tanh_derivative(net_h2)


w1 -= learning_rate * deltaH1 * i1
w2 -= learning_rate * deltaH1 * i2
w3 -= learning_rate * deltaH2 * i1
w4 -= learning_rate * deltaH2 * i2


w5 -= learning_rate * delta_o1 * out_h1
w6 -= learning_rate * delta_o1 * out_h2
w7 -= learning_rate * delta_o2 * out_h1
w8 -= learning_rate * delta_o2 * out_h2


print('\n\nBackpropagation')
print("Updated Weights:")
print(f"W1: {w1}, W2: {w2}, W3: {w3}, W4: {w4}")
print(f"W5: {w5}, W6: {w6}, W7: {w7}, W8: {w8}")