from keras.applications.inception_v3 import InceptionV3
from numpy.random.mtrand import shuffle
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K, Input
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Activation
from keras.optimizers import Adam, RMSprop
from numpy.random import shuffle
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Input

K.set_image_dim_ordering('tf')

def path_join (dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]


def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = r"True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()



def load_images(image_paths):
    # Load the images from disk.
    images = [plt.imread(train_dir) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)


def plot_training_history(history):
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']

    # Get it for the validation-set (we only use the test-set).
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss')

    # Plot it for the test-set.
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss')

    # Plot title and legend.
    plt.title('Training and Test Accuracy')
    plt.legend()

    # Ensure the plot shows correctly.
    plt.show()


train_dir = 'C:/tf_files/'
test_dir = r'C:\Users\Shangai\Desktop\PecuTest/'

input_tensor = Input(shape=(299, 299, 3))

base_model=InceptionV3(weights='imagenet',include_top= False,input_tensor=input_tensor)

input_shape=base_model.output_shape[1:]

model=base_model

datagen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=180,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=[0.9, 1.5],
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='nearest')


datagen_test = ImageDataGenerator(rescale=1./255)

batch_size = 20


generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=r'C:\augmented')


generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)

#number of steps
steps_test = generator_test.n / batch_size

steps_train = generator_train.n / batch_size


image_paths_train = path_join(train_dir, generator_train.filenames)
image_paths_test = path_join(test_dir, generator_test.filenames)

cls_train = generator_train.classes
cls_test = generator_test.classes
#class names
class_names = list(generator_train.class_indices.keys())
num_classes = generator_train.num_classes

# Get the true classes for those images.
cls_true = cls_train[1000:1009]

print(class_names)
print(num_classes)

class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)
print(class_weight)

transfer_layer = model.get_layer('mixed10')
print(transfer_layer.output)

conv_model = Model(inputs=model.input,
                   outputs=transfer_layer.output)

print(model.layers[0].input_shape)

for layer in conv_model.layers:
    layer.trainable = False

def print_layer_trainable():
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))

#TRANSFER LEARNING!!
# Start a new Keras Sequential model.


new_model = Sequential()

# Add the convolutional part of the VGG16 model from above.
new_model.add(conv_model)

# Flatten the output of the VGG16 model because it is from a
# convolutional layer.
new_model.add(Flatten())

# Add a dense (aka. fully-connected) layer.
# This is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(1024, activation='relu'))

# Add a dropout-layer which may prevent overfitting and
# improve generalization ability to unseen data e.g. the test-set.
new_model.add(Dropout(0.5))

# Add the final layer for the actual classification.
new_model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(lr=1e-5)

loss = 'categorical_crossentropy'

metrics = ['categorical_accuracy']

new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

epochs = 15
steps_per_epoch = 110

history = new_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)
