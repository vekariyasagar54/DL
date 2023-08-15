#imports
import torch
import torch.nn as nn #loss fun
import torch.optim as optim #adams
import torch.nn.functional as F #relu tanh
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#create fully connected network
class NN(nn.Module):
    def __init__(self , input_size , num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size , 50)
        self.fc2 = nn.Linear(50 , num_classes)

    def forward(self , x):            #we don't use this function is it neccessary to put here
        x = F.relu(self.fc1(x))      #is it like you calculated through linear function then whole output is passed into the relu fun!!
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, in_channels , num_classes):
        super(CNN , self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1 , out_channels= 8 , kernel_size=(3,3) , stride=(1,1) , padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2) , stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels = 8 , out_channels= 16 , kernel_size=(3,3) , stride=(1,1) , padding=(1,1))
        # self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.conv3 = nn.Conv2d(in_channels = 16 , out_channels= 32 , kernel_size=(3,3) , stride=(1,1) , padding=(1,1)) # 1% accuracy will inc using one more conv layer
        self.fc1 = nn.Linear(16*7*7 , num_classes)


    def forward(self , x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
       # x = F.relu(self.conv3(x))
       # x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x



# model = NN(784 , 10)
# x = torch.randn(64 , 784)
# print(model(x).shape)


#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

#load data
train_dataset = datasets.MNIST(root = 'dataset/'  ,train=  True , transform = transforms.ToTensor(),download = True)
# print(train_dataset)
train_loader = DataLoader(dataset = train_dataset , batch_size=batch_size , shuffle=True)
# print(train_loader)
test_dataset = datasets.MNIST(root = 'dataset/' , train=  False , transform = transforms.ToTensor(),download = True)
test_loader = DataLoader(dataset = test_dataset , batch_size=batch_size , shuffle=True)
# print(train_loader)

#init network
model = CNN(in_channels=in_channels , num_classes = num_classes).to(device)

#we can create sequential model like this faster
# model = nn.Sequential(
#     nn.Linear(input_size, 50),
#     nn.ReLU(),
#     nn.Linear(50, num_classes)
# )
#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters() , lr = learning_rate)


#train network
for epoch in range(num_epochs):
    for batch_idx , (data , targets) in enumerate(train_loader):
        data = data.to(device =device)
        targets = targets.to(device = device)

        #get to correct shape
        # data = data.reshape(data.shape[0] , -1)

        # forward
        scores = model(data)  #what is going on here : it is actually fun of nn.Module and that has function forward which has to be implemented by its subclasses
        loss = criterion(scores , targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient decent or adam step
        optimizer.step()


# model.train()

#check accuracy on traing and test to see how good model was?

def check_accuracy(loader , model):
    if loader.dataset.train:
        print("checking accuracy on training")
    else:
        print("checking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()      #it is droping unneccesary neurons so we can save that computation

    with torch.no_grad():
        for x , y in loader:
             x = x.to(device =device)
             y = y.to(device = device)
             # print('shape of the feature data' , x.shape ,  'shape of target data:' , y.shape)
             # x = x.reshape(x.shape[0] , -1)

             scores = model(x)
             _, prediction = scores.max(1)
             num_correct += (prediction == y).sum()
             num_samples +=  prediction.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

        model.train()


check_accuracy(train_loader , model)
check_accuracy(test_loader , model)
