import math 

inputs = [1,2,6]
outputs = [2,4,12]

weight = [1,1,1,1]
bias = [0,0,0]



total = [0,0,0] 

lr = 0.1

def sigmoid(x):
  return max(0,x)


def sigdev(x):
  return  (x > 0) * 1

for _ in range(10000):

  for x in range(3):
    
    total[0] = sigmoid(weight[0]* inputs[x]+ bias[0]) * weight[2] +sigmoid(weight[1]* inputs[x]+ bias[1]) *weight[3] + bias[2]

    error = math.pow(outputs[x]-total[0],2)
    bias[2] -= ( -2*(outputs[x]-total[0])) *lr
    weight[0] -=  (
      ( -2*(outputs[x]-total[0])  * weight[2] * sigdev(weight[0]* inputs[x]+ bias[0]) * inputs[x])
    ) * lr 
    bias[0] -=  (
      (-2*(outputs[x]-total[0]) * weight[2] * sigdev(weight[0]* inputs[x]+ bias[0]))
      
    )* lr 
    weight[1] -=  (
      ( -2*(outputs[x]-total[0]) * weight[3] * sigdev(weight[1]* inputs[x]+ bias[1]) * inputs[x])
      
    )*lr
    bias[1] -=  (
      ( -2*(outputs[x]-total[0]) * weight[3] * sigdev(weight[1]* inputs[x]+ bias[1]))
     
    )*lr
    weight[2] -= ( -2*(outputs[x]-total[0])) *sigmoid(weight[0]* inputs[x]+ bias[0])*lr
    weight[3] -= ( -2*(outputs[x]-total[0]))*sigmoid(weight[1]* inputs[x]+ bias[1]) *lr
    print(total[0])


print(sigmoid(weight[0]* 0.8+ bias[0]) * weight[2] +sigmoid(weight[1]* 0.8 + bias[1]) *weight[3] + bias[2])