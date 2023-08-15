import torch

########################################
#         init of tensor               #
########################################

device = "cuda" if(torch.cuda.is_available()) else "cpu"
my_ten = torch.tensor([[1,2] , [2,3]] , dtype=torch.float32 , device = device , requires_grad = True)
print(my_ten)
print(my_ten.dtype  , my_ten.device , my_ten.shape , my_ten.requires_grad)

#other common init for tensors

# x = torch.empty(size= (3,3))
# print(x)
# x = torch.zeros(size = (3,3))
# print(x)
# x = torch.rand(size = (3,3))
# print(x)
# x = torch.ones(size = (3,3))
# print(x)
# x = torch.eye(3,3)
# print(x)
# x = torch.arange(start = 0 , end = 5 ,step =1)
# print(x)
# x = torch.linspace(start=0.1 , end=1 , steps= 12)
# print(x)
# x = torch.empty(size = (2,3)).normal_(mean = 0 , std=1) #normal_ fill the tesnor with given specification with random vals everytime
# print(x)
# x = torch.empty(size = (2,3)).uniform_(0 ,1)#fill the tensor with the uniform random values within given range
# print(x)
# # tensor([[ 0.3158, -2.1586, -0.0101],
# #         [-1.3359,  0.0351, -0.2298]])
# # tensor([[0.9186, 0.2315, 0.8246],
# #         [0.5784, 0.8517, 0.7715]])
# x = torch.diag(torch.ones(3)) #make diagonal matrix


#how to init nd convert tensor to other types

tensor= torch.arange(4)
print(tensor)
print(tensor.bool())# 0/1
print(tensor.short())#int16
print(tensor.long()) #int64
print(tensor.half())#float16
print(tensor.float())#float32
print(tensor.double())#float64

#convert array to tensor

import numpy as np
arr = np.zeros((3,3))
tensor = torch.from_numpy(arr)
np_arr_back = tensor.numpy()
# print(arr  , tensor , np_arr_back)



