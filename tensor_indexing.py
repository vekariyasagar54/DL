import torch

###########################################
#          tensor indexing                #
###########################################

batch_size =10
features = 25
x = torch.eye(batch_size , features)

print(x[0]) # or use x[0,:]  , mean : you want to extract 1st batch features paras

print(x[:,0]) # extracting first feature paras of all batches

print(x[2, 0:10]) # it will give 2nd batch's features paras from 0 to 10(excluding)

x[0,0] = 100

#fancy indexing
x = torch.arange(10)
indices = [2,3,6]
print(x[indices])

x = torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x[rows, cols].shape)

#more advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])  #peaks the elements which is satisfied the condition given
print(x[x.remainder(2) == 0])

#useful ops

print(torch.where(x > 5 , x, x*2)) #if x >5 then make it x otherwise double the vals
print(torch.tensor([1,1,1,3,4,4]).unique())
print(x.ndimension()) # gives how many dimns it have
print(x.numel()) #count how many elements in tensor



