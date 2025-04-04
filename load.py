from mtcnn import MTCNN
import torch
import torch.nn
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image



class Load():

    def load_data(self, file_path):

        tensor = []
        size = (128, 128)

        
        for file_name in os.listdir(file_path):
            if file_name.endswith('jpg'):
                path = os.path.join(file_path, file_name)
                img = Image.open(path).convert('RGB')
                img = img.resize(size=size)
                img_array = np.array(img) / 255.0

                
                '''
                Permute required as we need it in HWC format (channels, height, width)
                PyTorch Expectation: PyTorch models (like convolutional neural networks) expect image data in the (C, H, W) format, where:
                    C is the number of channels (3 for RGB images),
                    H is the height (128 in this case),
                    W is the width (128 in this case).
                '''

                img_array = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1) 

                img_array = img_array.reshape(-1)

                tensor.append(img_array)


        return torch.stack(tensor)




