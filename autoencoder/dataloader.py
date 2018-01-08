import os
import numpy as np
import random
from skimage import io
from skimage.transform import resize
import time

class Dataset:
    def __init__(self, data_path):
        print('Load {0} ...' .format(data_path))
        self.data_path= data_path
        self.data_list = [os.path.join(data_path, file_path) for file_path in os.listdir(data_path)]
        self.length_data = len(self.data_list)
        random.seed(time.time())
        random.shuffle(self.data_list)
        print('Load {0} SUCCESS' .format(data_path))
        
        self.epoch = 0
        
    def load_batch(self, batch_size, idx, is_crop=True, is_resize=True, size=[96, 96]):
        batch_data = []
        batch_list = []    
        
        if idx + batch_size >= self.length_data:
            cut_point = idx + batch_size - self.length_data + 1
            batch_list = self.data_list[idx:-1]
            random.shuffle(self.data_list)
            batch_list += self.data_list[:cut_point]
            idx = cut_point 
            self.epoch += 1
        else:
            batch_list = self.data_list[idx:idx+batch_size]
            idx += batch_size
            
        for file_path in batch_list:
            img = io.imread(file_path)
            if is_crop:
                short_edge = min(img.shape[:2])
                yy = int((img.shape[0] - short_edge) / 2)
                xx = int((img.shape[1] - short_edge) / 2)
                img = img[yy: yy + short_edge, xx: xx + short_edge]
            if is_resize:
                img = resize(img, size)   
            batch_data.append(img)
        batch_data = np.array(batch_data)

        return self.epoch, idx, batch_data
