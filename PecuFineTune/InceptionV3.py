from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
import os.path
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from keras.optimizers import Adam

# load the inception v3 model
base_model = InceptionV3(weights='imagenet', include_top=False)


#defining the input size (specifically for inception)
img_width, img_height = 299, 299

#Saving chekpoints of the weights
top_layers_checkpoint_path = 'cp.top.best.hdf5'
fine_tuned_checkpoint_path = 'cp.fine_tuned.best.hdf5'
new_extended_inception_weights = 'final_weights.hdf5'

#Data sets folders
train_data_dir = r'C:\data\train'
validation_data_dir = r'C:\data\validation'
save_path = r'C:\data\TrainHistory'

#setting the number of images in training and validation.
nb_train_samples = 1700
nb_validation_samples = 189

#How much iterations do we want
top_epochs = 120
fit_epochs = 120
batch_size = 15


##FINE TUNING THE MODEL

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer with a rectifier activation
# "It has been demonstrated for the first time in 2011 to enable better training of deeper networks"
# -Xavier Glorot, Antoine Bordes and Yoshua Bengio (2011). Deep sparse rectifier neural networks
x = Dense(1024, activation='relu')(x)
# and here we tell the model that we have 5 classes (araña, abeja, chinche, hormiga, mosquito)
predictions = Dense(5, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# load the checkpoint if it exists, to resume training
if os.path.exists(top_layers_checkpoint_path):
	model.load_weights(top_layers_checkpoint_path)
	print ("Checkpoint '" + top_layers_checkpoint_path + "' loaded.")

# first we train only the top layers (which were randomly initialized)
# And we freeze all convolutional InceptionV3 layers, so the pre-trained model don't start training from 0 again (it would take ages tho)
for layer in base_model.layers:
    layer.trainable = False

# compile the model, we have to do this every time before training
# We use adam optimizer because it has proven better results
# and a categorical crossentropy to classify more than 2 classes (instead of binary_crossentropy)
# And we tell the model to learn from the accuracy in the predictions
model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'], )

# Here we use the keras data generator function, which allow us to get the most of our training set
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


# Save the model after every epoch.
mc_top = ModelCheckpoint(top_layers_checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# Save the TensorBoard logs
# **Unfortunately, tensorboard doesn´t work with the keras data generators, so we set the frequence to 0, it means it will not write.
tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

#we define some variables from the generators that we made
cls_train = validation_generator.classes
cls_test = validation_generator.classes

#And after some terrible inaccuracy in the predictions, i decided that i needed to balance the weights in the classes
#I tried doing this with Sklearn compute weights function. But... There was no improvement.
class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)


#This is to set to know what is the "array" that corresponds to each class.
class_names = list(train_generator.class_indices.keys())

print(class_names)
print(class_weight)


# train the model on the new data for a few epochs

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=top_epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples // batch_size,
    class_weight=class_weight,
    callbacks=[mc_top])

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.



#Save the model after every epoch.
mc_fit = ModelCheckpoint(fine_tuned_checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

#This is the checkpoint loader for the finetuning
if os.path.exists(fine_tuned_checkpoint_path):
	model.load_weights(fine_tuned_checkpoint_path)
	print ("Checkpoint '" + fine_tuned_checkpoint_path + "' loaded.")

#I decided to let this as Keras shows in the documentation.
# we chose to train the top 2 inception blocks, we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True


# we need to recompile the model for these modifications to take effect
# Again we are using adam optimizer, with a very low learning rate.
model.compile(optimizer=Adam(lr=0.0005, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=fit_epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples // batch_size,
    class_weight=class_weight,
    callbacks=[mc_fit])

#We save the weights of the model
model.save_weights(new_extended_inception_weights)


#And save the model to a json, if its needed
model_json = model.to_json()
with open("modelv3.json", "w") as json_file:
    json_file.write(model_json)

#And we save again the model as a h5 for usage in the other code
model.save('pecuscope_modelv3.h5')

#This is the version 3 of the pecuscope model. Which didn't show much of improvement, so we kept the first one.

