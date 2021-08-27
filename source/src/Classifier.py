import keras
import numpy as np
from numpy import asarray
import PIL
from PIL import Image, ImageOps
import os
import io

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Classifier:
    
    model = None
    LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    def __init__(self) -> None:
        self.model = keras.models.load_model('SignModel2.h5')

    def predict_sign(self, image_bytes):
        #
        image = Image.open(io.BytesIO(image_bytes))  
        
        #Crop the image so that new image is a squre that shares the midpoint with the un-cropped image
        width, height = image.size
        print(width, height)

        if height < width:
            top = 0
            bottom = height
            left = int((width/2) - (height/2))
            right = int((width/2) + (height/2))
            image = image.crop((left, top, right, bottom))

        elif height > width:
            top = int((height/2) - (width/2))
            bottom = int((height/2) + (width/2))
            left = 0
            right = width
            image =  image.crop((left, top, right, bottom))

         
        #Resize and grayscale
        image = image.convert('L')
        # image.save('testingGrayscale.png')

        image = image.resize(size = (28, 28))
        # image.save('testing.png')

        # print(image.size)

        #Normalize
        image_data = np.array(image)
        # print(image_data.shape)
        image_data = image_data.astype(np.float)
        image_data /= 255.

        #Reshape to put into the model
        #image_data = image_data[:, :, 0]
        image_data = image_data[np.newaxis, ..., np.newaxis]
        
        try:
            image_data = image_data[:, :, :, 1]
        except:
            # print('ee')
            pass

        #Make prediction
        prediction = self.model.predict(image_data)

        #Find the letter
        index = np.argmax(prediction)
        ans = self.LABELS[index]
        # print(prediction)
        # print(ans)

        return ans




'''
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
            'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
            'w', 'x', 'y', 'z']


image = image[..., np.newaxis]
image = image[np.newaxis, ...]
print(image.shape)

prediction = model.predict(image)

index = np.where(1, prediction)
ans = LABELS[index]
print(ans)
'''