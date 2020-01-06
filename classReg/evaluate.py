import imageio
import numpy as np
from keras.models import model_from_json

#load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


from PIL import Image
from resizeimage import resizeimage

fd_img = open('../fewshot4/generate_5_8857_5_38855.png', 'rb')
#fd_img = imageio.imread('generate_10829.png')
img = Image.open(fd_img)
img.show()
#print(np.asarray(img))
resized_img = np.asarray(resizeimage.resize_contain(img, [28, 28]))
resized_img = resized_img[ :, :, 0]
#img.save('generate_10829_28.png', img.format)
fd_img.close()

#test_img = imageio.imread('generate_10829_28.png')

print('original size\n')
#print(fd_img.shape)
print('resized\n')
print(resized_img.shape)
#print(resized_img)
pred = loaded_model.predict(resized_img.reshape(1, 28, 28, 1))
print(pred)
