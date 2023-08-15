import torch
####################################
#    tensor reshaping              #
####################################

x = torch.arange(9)

x_3x3 = x.view(3,3) #view can only works in contigueous memory
print(x.shape)
x_3x3 = x.reshape(3,3)#it can work on any matrix but is costly bcz it does coping things
print(x.shape)
y = x_3x3.t() #transpose of x33
print(y)
# print(y.view()) gives you error as y transpose is not contigeous in memory
#use instead
print(y.contiguous().view(9))
print(y.reshape(9))

x1 = torch.rand(2,3)
x2 = torch.rand(2,3)

print(torch.cat((x1,x2) , dim = 0)) #concat the tensor about given dim

z = x1.view(-1)#-1 mean you want to flatten everything
x = torch.rand((32, 2, 4))
z = x.view(32,-1) #keeping the 32 as it is i want to faltten out anything else
print(z.shape)

z = x.permute(0,2,1) # you want to change the dimns like her 0->0 and 1st and 2nd dimns are swapped
print(z.shape) # 32 4 2


x = torch.arange(10)
print(x.unsqueeze(0).shape) #change the dimns
print(x.unsqueeze(1).shape)
print(x.unsqueeze(1).unsqueeze(1).shape)
print(x.unsqueeze(1).squeeze(1).shape)

