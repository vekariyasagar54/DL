import torch

##########################################
#                tensor math & com ops   #
##########################################

x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])

#add
z1 = torch.empty(3)
torch.add(x , y , out=z1)
print(z1)

z2 = torch.add(x ,y)
print(z2)

z = x + y
print(z)

print(x + y)

#subtraction

z = x - y

#division

z = torch.true_divide(x ,y)

#inplace ops
t = torch.zeros(3)
t.add_(x) #if any fun name is follwed by _(underscore) then it is inpalce ops
t += x # it is also inpalce it is faster bcz it doesnot create extra memory for it

# if you do like t = t + x , then compiler may store the temp values of t+x computattion to other memory then assign back to t
#if you do like t += x , then you will let it know that you want to store the value of t +x to back in t , so it is bit optimized to use

#exponentitation

z = x.pow(2)
z = x ** 2

#simple comparision
z = x > 8
z = x < 8

#matrix multi

x1 = torch.rand(size=(2,3))
x2 = torch.rand(size=(3,5))
x3 = torch.mm(x1 , x2)
x3 = x1.mm(x2)

#matrix expoenitation

m = torch.rand((5,5))
print(m.matrix_power(3)) # out the matrix a = m*m*m  , three times mm

#elementwise multi
z = x * y
print(z)

#dot product
z = torch.dot(x ,y)

#batch mm
#refer this for more details : https://christopher5106.github.io/deep/learning/2018/10/28/understand-batch-matrix-multiplication.html
batch = 32
n = 10
m = 20
p = 30

t1 = torch.rand((batch , n , m)) # 3 dimension mm
t2 = torch.rand((batch , m , p))
out = torch.bmm(t1 , t2)

#broADCASTING

x1 =torch.ones(size = (2,2))
x2 = torch.rand((1,2))
# smaller dims expands to match bigger dimns and then computations happen

z = x1 -x2
z = x1 ** x2

#other ops
sum_x = torch.sum(x1,dim=0) # reduce the dimns which is mentioned
print(sum_x)
val , ind = torch.max(x , dim= 0)
z = torch.argmax(x , dim = 0)#only retuen index
z = torch.abs(x)
mean_x = torch.mean(x.float() , dim = 0)
z = torch.eq(x , y) #to check two tensors are same or not
sorted_y , indices = torch.sort(y , dim = 0, descending=False) #return indcies which is swapped

z = torch.clamp(x , min = 0 , max = 10) #if vals in x is <0 then it is set to be 0 and if >10 then set to be 10
x = torch.tensor([1,1,1,0,1,1] , dtype= torch.bool)
z = torch.any(x) #return true if any val is true
print(z)
z = torch.all(x)#return true if all vals is true
print(z)




