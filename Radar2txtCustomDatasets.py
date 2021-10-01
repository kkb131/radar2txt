# In[]
from torch.utils.data import Dataset
import numpy as np
import cv2

# In[]

class custom_dataset(Dataset):
    def __init__(self, data): 
        self.data = data

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,idx):       
        data = self.data[idx]
        
        # X = tmp["X_train"]
        # Y = tmp["Y_feat"]
        #Y = tmp["Y_hidden"]

        return data