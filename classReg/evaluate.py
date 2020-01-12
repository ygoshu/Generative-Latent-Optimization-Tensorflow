import imageio
import numpy as np
from keras.models import model_from_json
from PIL import Image
from resizeimage import resizeimage

def load_classifier(model_json, model_h5):
    #load json and create model
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_h5)
    print("Loaded model from disk")
    return loaded_model

def get_class_prob_form_file(loaded_model, files):
    for file_n in files:
        print('reading image')
        fd_img = imageio.imread(file_n)
        img = Image.fromarray(fd_img)
        resized_img = np.asarray(resizeimage.resize_thumbnail(img, [28, 28]))
        print('image resized')

        resized_img = resized_img.astype('float32') 
        resized_img = resized_img/255
        pred = loaded_model.predict(resized_img.reshape(1,28, 28, 1)) 
        print(file_n)
        print(pred.argmax())
        print(pred[0][pred[0].argmax()])



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_json', type=str)
    parser.add_argument('--model_h5', type=str)
    parser.add_argument('--files', nargs='+', type=str)
    config = parser.parse_args()
    print('loaded_model')
    loaded_model = load_classifier(config.model_json, config.model_h5)
    print('getting probs')
    get_class_prob_form_file(loaded_model, config.files)
if __name__== "__main__" :
    main()
