#type:ignore
from torch.utils.data import Dataset,DataLoader,SequentialSampler
import cv2
from PIL import Image
import pandas as pd
from torchvision.transforms import transforms
import torch
import numpy as np
import time

cuda=torch.cuda.is_available()

class ImageDataset(Dataset):
    def __init__(self,csv_path,image_path,transform=None):
        self.data=pd.read_csv(csv_path)
        self.image_path=image_path
        self.transform=transform
    
    def __getitem__(self,index):
        
        image=cv2.imread(self.image_path+self.data.iloc[index,0])
        image=Image.fromarray(image)
        if self.transform:
            image=self.transform(image)
        return image
    
    def __len__(self):
        return len(self.data)

transform=transforms.Compose([
    transforms.Resize((256,156)),
    transforms.ToTensor()
])
batch_size=100
dataset=ImageDataset('./labels.csv','./Database/rgb/',transform=transform)
sampler=SequentialSampler(list(range(len(dataset))))
kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
dataloader=DataLoader(dataset=dataset,batch_size=batch_size,sampler=sampler,**kwargs)

pop_mean=[]
pop_std0=[]
pop_std1=[]
start_time=time.time()
for idx,data in enumerate(dataloader):
    data=data.numpy()
    batch_mean=np.mean(data,axis=(0,2,3))
    batch_std0=np.std(data,axis=(0,2,3))
    batch_std1=np.std(data,axis=(0,2,3),ddof=1)
    pop_mean.append(batch_mean)
    pop_std0.append(batch_std0)
    pop_std1.append(batch_std1)
    if idx%(int(len(dataloader)/10))==0:
        print(f'Batch: {idx}/{len(dataloader)} Time passed: {time.time()-start_time}')
pop_mean=np.array(pop_mean).mean(axis=0)
pop_std0=np.array(pop_std0).mean(axis=0)
pop_std1=np.array(pop_std1).mean(axis=0)
print(f'mean: {pop_mean},std0: {pop_std0}, std1:{pop_std1}')