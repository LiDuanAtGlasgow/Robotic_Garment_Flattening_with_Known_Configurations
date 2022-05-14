#type:ignore
from __future__ import print_function
import argparse
from pickletools import optimize
from unicodedata import category

from zmq import device
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
import pandas as pd
import os
import time
import cv2
import torchvision.models as models
from torch.optim import lr_scheduler
from sklearn.preprocessing import OneHotEncoder

enc=OneHotEncoder(handle_unknown='ignore')
categories=[['jean'],['shirt'],['sweater'],['tshirt'],['towel']]
enc.fit(categories)


class Dataset_(Dataset):
    def __init__(self,csv_path,image_address,transform):
        self.data_csv=pd.read_csv(csv_path)
        self.path=image_address
        self.transform=transform
    
    def __getitem__(self,index):
        image=cv2.imread(self.path+self.data_csv.iloc[index,0],0)
        label=self.data_csv.iloc[index,1]
        if self.transform is not None:
            image=self.transform(image)
        shape=self.data_csv.iloc[index,2]
        shape=enc.transform([[shape]]).toarray()
        shape=shape.astype(int)
        #print ('shape:',shape)
        return image,label,shape
    
    def __len__(self):
        return len(self.data_csv)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*126*126, 128)
        self.fc2=nn.Linear(5,76)
        self.fc3 = nn.Linear(204, 50)

    def forward(self, x,shape):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        shape=self.fc2(shape)
        x=torch.cat([x,shape],dim=1)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

def frozon (model):
    for param in model.parameters():
        param.requires_grad=False
    return model

class ResNet18(nn.Module):
    def __init__(self) -> None:
        super(ResNet18,self).__init__()
        modeling=frozon(models.resnet18(pretrained=True))
        modules=list(modeling.children())[:-2]
        self.features=nn.Sequential(*modules)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc0=nn.Sequential(
            nn.Linear(512*8*8,256),
            nn.PReLU()
        )
        self.fc1=nn.Sequential(
            nn.Linear(332,332),
            nn.PReLU()
        )
        self.fc2=nn.Sequential(
            nn.Linear(5,76),
            nn.PReLU()
        )
        self.fc3=nn.Sequential(
            nn.Linear(332,50),
            nn.PReLU()
        )

    def forward(self,x,shape):
        output=self.features(x)
        output=self.fc0(output.reshape(output.shape[0],-1))
        shape=self.fc2(shape)
        shape=shape.reshape(shape.shape[0],-1)
        output=torch.cat([output,shape],dim=1)
        output=self.fc3(self.fc1(output))
        output = F.log_softmax(output, dim=1)
        return output
    
    def get_emdding(self,x):
        return self.forward(x)


def train(args, model, device, train_loader, optimizer, epoch,k):
    model.train()
    for batch_idx, (data, target,shape) in enumerate(train_loader):
        data, target, shape = data.to(device), target.to(device),shape.type(torch.float32).to(device)
        optimizer.zero_grad()
        output = model(data,shape)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('[k-value] {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(k+1,
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader,k,image_format='depth'):
    model.eval()
    test_loss = 0
    correct = 0
    '''
    jean_acc=0
    jean_cor=0
    shirt_acc=0
    shirt_cor=0
    sweater_acc=0
    sweater_cor=0
    towel_acc=0
    towel_cor=0
    tshirt_acc=0
    tshirt_cor=0
    '''
    with torch.no_grad():
        for _, (data, target,shape) in enumerate(test_loader):
            data, target, shape = data.to(device), target.to(device),shape.type(torch.float32).to(device)
            output = model(data,shape)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            '''
            for i in range(len(pred)):
                if pred[i]<10:
                    if pred[i]==target[i]:
                        towel_cor+=1
                    towel_acc+=1
                elif 10<=pred[i]<20:
                    if pred[i]==target[i]:
                        tshirt_cor+=1
                    tshirt_acc+=1
                elif 20<=pred[i]<30:
                    if pred[i]==target[i]:
                        shirt_cor+=1
                    shirt_acc+=1
                elif 30<=pred[i]<40:
                    if pred[i]==target[i]:
                        sweater_cor+=1
                    sweater_acc+=1
                else:
                    if pred[i]==target[i]:
                        jean_cor+=1
                    jean_acc+=1
            '''

    test_loss /= len(test_loader.dataset)

    print('\n [k-value] {} Validate set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        k+1,test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    '''
    print ('\n -------------------------------------------------')
    print ('\n Image Format:',image_format)
    print ('\n [towel]:',100*(towel_cor/towel_acc),'%')
    print ('\n [tshirt]:',100*(tshirt_cor/tshirt_acc),'%')
    print ('\n [shirt]:',100*(shirt_cor/shirt_acc),'%')
    print ('\n [sweater]:',100*(sweater_cor/sweater_acc),'%')
    print ('\n [jean]:',100*(jean_cor/jean_acc),'%')
    print ('\n -------------------------------------------------')
    '''

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Known Configurations Project')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--image_format',type=str,default='depth',help='image format')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    if args.image_format=='depth':
        normalises=[0.02428423,0.02427759,0.02369768,0.02448228]
        stds=[0.0821249,0.08221505,0.08038522,0.0825848]
    elif args.image_format=='rgb':
        normalises=[0.04105412,0.03650091,0.03348222,0.0361118]
        stds=[0.132038,0.11855838,0.11531012,0.12060914]
    else:
        print ('wrong image format, quitting...')
        exit()

    k=4
    for index in range(k):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.Normalize((normalises[index],), (stds[index],))
            ])
        image_address='./Database/'+args.image_format+'/'
        train_csv='./collection_of_trains/'+str(index+1)+'/train.csv'
        val_csv='./collection_of_vals/'+str(index+1)+'/val.csv'
        test_csv='./collection_of_tests/'+str(index+1)+'/test.csv'
        train_dataset=Dataset_(csv_path=train_csv,image_address=image_address,transform=transform)
        val_dataset=Dataset_(csv_path=val_csv,image_address=image_address,transform=transform)
        test_dataset=Dataset_(csv_path=test_csv,image_address=image_address,transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
        test_loader=torch.utils.data.DataLoader(test_dataset, **test_kwargs)

        model = ResNet18().to(device)
        model=model.float()
        #model.load_state_dict(torch.load('./Model/'+args.image_format+'/'+str(index+1)+'/KCNet_'+args.image_format+'_'+str(index+1)+'.pt'))
        #model.eval()
        params=[]
        print ('\n ----------------Params---------------------------')
        for name,param in model.named_parameters():
            if param.requires_grad==True:
                print ('name:',name)
                params.append(param)
        print ('\n -------------------------------------------------')
        #optimizer = optim.Adadelta(params, lr=args.lr)
        #optimizer=optim.Adadelta(model.parameters(), lr=args.lr)
        optimizer=optim.Adam(params,lr=args.lr)
        scheduler=StepLR(optimizer,8,gamma=0.1,last_epoch=-1)
        #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch,index)
            test(model, device, val_loader,index)
            scheduler.step()


        test(model,device=device,test_loader=test_loader,k=index,image_format=args.image_format)
    
        model_file='./Model/'+args.image_format+'/'+str(index+1)+'/'
        if not os.path.exists(model_file):
            os.makedirs(model_file)
        torch.save(model.state_dict(), './Model/'+args.image_format+'/'+str(index+1)+'/KCNet_%f.pt'%time.time())


if __name__ == '__main__':
    main()