'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
import h5py
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Dense, ZeroPadding2D, Conv2D, MaxPooling2D,BatchNormalization
from keras.models import Model
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

batch_size = 400
num_classes = 10
epochs = 400
data_augmentation = False
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_multiloss_model.h5'

def main_loss(y_true,y_pred):
        ind=0
        if(ind==0):
            arr = np.load('labels.npy')
            arr = tf.convert_to_tensor(arr,dtype='float32')
            sh = tf.shape(y_pred)
            vec1 = tf.tensordot(y_pred,arr,axes=[1,0])
            vec2 = tf.tensordot(y_true,arr,axes=[1,0])
            ans = tf.square(vec1-vec2)

        return ans

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y1), (x_test, y_t1) = cifar10.load_data()

'''l=[]
for i in range(y_train.shape[0]):
    if(y_train[i]==3):
        l.append(i)
print('here')


x_train=np.delete(x_train,l,axis=0)
y_train=np.delete(y_train,l,axis=0)
y1=np.delete(y1,l,axis=0)
print(x_train.shape)
print('here')'''


arr = np.load('labels.npy')

y = np.zeros((y1.shape[0],300))

y_t = np.zeros((y_t1.shape[0],300))


for i in range(y.shape[0]):
    y[i,:] = arr[int(y1[i])]
   

for i in range(y_t.shape[0]):
    y_t[i,:] = arr[int(y_t1[i])]

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

inputs = Input(shape=(32,32,3))

x = (Conv2D(32, (3, 3), activation='relu'))(inputs)
x = Conv2D(64, (3, 3), activation='relu')(x)
x =(MaxPooling2D((2,2), strides=(2,2)))(x)
x = (Dropout(0.5))(x)



x = (Conv2D(128, (3, 3), activation='relu'))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
x =(MaxPooling2D((2,2), strides=(2,2)))(x)
x = (Dropout(0.5))(x)

x = Conv2D(256, (1,1), activation='relu')(x)
x = Conv2D(256, (1,1), activation='relu')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
x = (Flatten())(x)
x = (Dense(512, activation='relu'))(x)
x = (Dropout(0.5))(x)
fine = (Dense(10, activation='softmax'))(x)

word2vec = (Dense(300, activation='softmax'))(fine)


model = Model(inputs=[inputs], outputs=[fine,word2vec])


# initiate RMSprop optimizer
#opt = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
#opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)


# Let's train the model using RMSprop
model.compile(loss=['categorical_crossentropy','mse'],
              optimizer=opt,
              metrics=['accuracy'],loss_weights=[1,0.2])


if not data_augmentation:
    print('Not using data augmentation.')
    cb = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                write_graph=True, write_images=True)
    history = model.fit(x_train, [y_train,y],
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              shuffle=True,verbose=2,callbacks = [cb])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # list all data in history
    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['dense_2_acc'])
    plt.plot(history.history['val_dense_2_acc'])
    plt.title('Model classification accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    #summarize for other loss
    plt.plot(history.history['dense_2_loss'])
    plt.plot(history.history['val_dense_2_loss'])
    plt.title('Model regression loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    pred1,pred2 = model.predict(x_test,batch_size=100)
    np.save('pred1.npy',pred1)
    np.save('pred2.npy',pred2)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),epochs=epochs,
                        validation_data=(x_test, y_test),
                        steps_per_epoch=x_train.shape[0] // batch_size)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.

scores = model.evaluate(x_test, [y_test,y_t], verbose=1)
print('Scores:', scores)
