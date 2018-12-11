from keras.models import load_model
import numpy as np
from keras.preprocessing import image

#we load the model, that must be in the same folder
model = load_model('pecuscope_model.h5')

#we set the size of our images
img_width, img_height = 299, 299

#we compile the mode before use with adam optimizer
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

#we define the image to predict
#then we convert it to array with numpy
#we add one dimension, so the tensor is 4D as our model input
img = image.load_img('test1.jpg', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

#we input the tensor and predict it.
#and we tell it to just give us one number (instead of 5 in array form)
#Depending on that number, we will get a result in a string form (mosquito, araña, etc..)
y_prob = model.predict(x, batch_size=None, verbose=0, steps=None)
print(y_prob)
y_classes = y_prob.argmax(axis=-1)
print(y_classes)
if y_classes==0:
    print("abeja")
if y_classes==1:
    print("araña")
if y_classes==2:
    print("chinche")
if y_classes==3:
    print("hormiga")
if y_classes==4:
    print("mosquito")


