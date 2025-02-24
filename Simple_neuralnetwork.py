import random
def tanh(x):
    exp_plus_x = 2.718281828459045 ** x
    exp_neg_x = 2.718281828459045 ** -x
    return (exp_plus_x - exp_neg_x) / (exp_plus_x + exp_neg_x)
def random_weight():
    return random.uniform(-.5,.5)
def weighted_sum (inputs,weigths, bias):
    return sum(i * w for i,w in zip(inputs,weigths))+bias


w1,w2,w3,w4 = random_weight(),random_weight(),random_weight(),random_weight()
w5,w6,w7,w8 = random_weight(),random_weight(),random_weight(),random_weight()

i1,i2 = .05,.1
b1,b2 =.5 , .7

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

print(f"prob of classs o1 : {out_o1}")
print(f"prob of classs o2 : {out_o2}")
print("It is class o1") if (out_o1 > out_o2) else print("It is class o2")
