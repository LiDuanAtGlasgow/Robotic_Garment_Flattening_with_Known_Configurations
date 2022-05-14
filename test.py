#type:ignore
from __future__ import print_function
import cv2
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import pandas as pd
import os
import time
import torchvision.models as models
from sklearn.preprocessing import OneHotEncoder

enc=OneHotEncoder(handle_unknown='ignore')
categories=[['jean'],['shirt'],['sweater'],['tshirt'],['towel']]
enc.fit(categories)

torch.manual_seed(1)

parser = argparse.ArgumentParser(description='Known Configurations Project')
parser.add_argument('--model_no',type=int,default=100,help='model number')
args = parser.parse_args()

class Get_Images():
    def __init__(self,image,shape,transforms=None):
        self.image=image
        self.transform=transforms
        self.shape=shape
    
    def __getitem__(self):
        image=self.image
        if not self.transform == None:
            image=self.transform(image)
        image=torch.unsqueeze(image,dim=0)
        shape=enc.transform([[self.shape]]).toarray()
        shape=shape.astype(int)

        return image,shape
    
    def __len__(self):
        return len(self.image)

def frozon (model):
    for param in model.parameters():
        param.requires_grad=False
    return model

class KCNet(nn.Module):
    def __init__(self) -> None:
        super(KCNet,self).__init__()
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


CATEGORIES=['towel','tshirt','shirt','sweater','jean']

def test(kcnet,data,shape,true_label,correct,acc,category,position_index):
    output=kcnet(data,shape)
    #print ('output:',output)
    pred=output.argmax(dim=1,keepdim=True)
    print ('true_postion:',category,position_index+1)
    print('predicted_postion:',CATEGORIES[pred.item()//10],pred.item()%10+1)
    if true_label==pred.item():
        correct+=1
    acc+=1
    return pred,acc,correct

if args.model_no==100:
    print ('You must assign a model number! quitting...')
    exit()
kcnet=KCNet()
kcnet.load_state_dict(torch.load('./Model/depth/'+str(args.model_no)+'/KCNet_depth_'+str(args.model_no)+'.pt'))
kcnet.eval()

normalises=[0.02428423,0.02427759,0.02369768,0.02448228]
stds=[0.0821249,0.08221505,0.08038522,0.0825848]
category='tshirt'
num_positions=1
position_index=9
num_frames=1
shape='tshirt'

acc=0
correct=0
for position in range(num_positions):
        for frame in range (num_frames):
            if category=='towel':
                category_index=0
            elif category=='tshirt':
                category_index=1
            elif category=='shirt':
                category_index=2
            elif category=='sweater':
                category_index=3
            elif category=='jean':
                category_index=4
            else:
                print ('category',category,'does not exit, quit...')
                break
            if num_positions==1:
                images_add='./test_images/'+category+'/pos_'+str(position_index+1).zfill(4)+'/'+str(frame+1).zfill(4)+'.png'
                true_label=category_index*10+position_index
            else:
                images_add='./test_images/'+category+'/pos_'+str(position+1).zfill(4)+'/'+str(frame+1).zfill(4)+'.png'
                true_label=category_index*10+position
            #print ('image_add',images_add)
            images=cv2.imread(images_add,0)
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256,256)),
                transforms.Normalize((normalises[args.model_no-1],), (stds[args.model_no-1],))
            ])
            data,shape=Get_Images(image=images,shape=shape,transforms=transform).__getitem__()
            shape=torch.from_numpy(shape).type(torch.float32)
            if num_positions==1:
                pred,acc,correct=test(kcnet,data,shape,true_label,correct,acc,category,position_index)
            else:
                pred,acc,correct=test(kcnet,data,shape,true_label,correct,acc,category,position)
accuracy=100*(correct/acc)
if num_positions !=1:
    print ('[category]',category,'[accuracy]',accuracy,'%')
else:
    if accuracy==0:
        print ('Known Configuration Recognition Is  Failed!')
    else:
        print ('Known Configuration Recognition Is Successful!')
print ('complete!')

