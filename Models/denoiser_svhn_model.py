from keras.layers import Dense, Input, BatchNormalization
from keras.layers import Conv2D, Flatten, MaxPool2D
from keras.layers import Reshape, Conv2DTranspose, Dropout
from keras.models import Model, Sequential
from keras import backend as K
import numpy as np
import tensorflow as tf

# define denoiser architecture
class ConvDenoiser():
    def __init__(self, X_noisy, X_clean, y_noisy, y_clean):
        self.X_noisy = X_noisy
        self.X_clean = X_clean
        self.y_noisy = y_noisy
        self.y_clean = y_clean

    def model(self, X_noisy, X_clean, y_noisy, y_clean):
        X_noisy = np.array(X_noisy)
        X_clean = np.array(X_clean)

        y_noisy = tf.keras.utils.to_categorical(y_noisy, 10)
        y_clean = tf.keras.utils.to_categorical(y_clean, 10)

        # reshape image arrays to match input shape of the autoencoder
        X_noisy = X_noisy.reshape((-1, 28, 28, 1))
        X_clean = X_clean.reshape((-1, 28, 28, 1))

        # normalize pixel values to the range [0, 1]
        X_noisy = X_noisy / 255.0
        X_clean = X_clean / 255.0

        # network params
        input_shape = (28, 28, 1)
        batch_size = 64
        validation_split = 0.25
        epochs = 25
        kernel_size = 6
        latent_dim = 64
        layer_filters = [64, 256]

        # autoencoder
        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs

        # Stack of Conv2D(64)-Conv2D(128)
        for filters in layer_filters:
            x = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=2,
                       activation='relu',
                       padding='same')(x)

        shape = K.int_shape(x)

        # latent vector
        x = Flatten()(x)
        latent = Dense(latent_dim, name='latent_vector')(x)

        # encoder
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # decoder
        latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
        x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)
        
        # stack of Conv2DTranspose(64)-Conv2DTranspose(32)
        for filters in layer_filters[::-1]: # last layer
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=2,
                                activation='relu',
                                padding='same')(x)
            
        x = Flatten()(x)
        outputs = Dense(10, activation='sigmoid', name='decoder_output')(x)
        
        # instantiate decoder
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # instantiate autoencoder
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        autoencoder.summary()

        # train
        history = autoencoder.fit(X_clean, y_clean,
                                validation_data=[X_noisy, y_noisy],
                                epochs=epochs,
                                batch_size=batch_size)

        # score
        loss, acc = autoencoder.evaluate(X_noisy, y_noisy, verbose=0)

        autoencoder.save('denoising_autoenconder.model')

        # predict autoencoder output
        predictions = autoencoder.predict(X_noisy)

        return history, loss, acc, predictions

# convolutional neural network model for SVHN dataset
class CNN():
    def __init__(self, X, X_test, y, y_test):
        self.X = X
        self.X_test = X_test
        self.y = y
        self.y_test = y_test

    def get_callbacks(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(mode="min", patience=5)
        return early_stopping

    def model(self, X, X_test, y, y_test):
        X = np.array(X)
        X_test = np.array(X_test)
        y = np.array(y)  
        y_test = np.array(y_test) 

        # y = tf.keras.utils.to_categorical(y, 10)
        # y_test = tf.keras.utils.to_categorical(y_test, 10)

        X = X.reshape((-1, 32, 32, 3))
        X_test = X_test.reshape((-1, 32, 32, 3))

        # normalize pixel values to the range [0, 1]
        X = X / 255.0
        X_test = X_test / 255.0

        # network params
        input_shape = (32, 32, 3)
        batch_size = 64
        validation_split = 0.1
        epochs = 25

        model = Sequential([

            Conv2D(32, (5,5), activation='relu', padding="same", input_shape=input_shape),
            BatchNormalization(),
            Conv2D(64, (5,5), activation='relu', padding="same"),
            
            Dropout(0.2),
            MaxPool2D((2,2), strides=2),
            
            Dense(32, activation='relu'),
            Flatten(),
            Dense(10, activation='softmax')

        ])

        model.summary()

        model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['acc'])

        early_stopping = self.get_callbacks()

        history = model.fit(X, y,
                    validation_split=validation_split,
                    epochs=epochs,
                    callbacks=early_stopping,
                    batch_size=batch_size)
        
        # predict autoencoder output
        y_preds = model.predict(X_test)

        # score
        loss, acc = model.evaluate(X_test, y_test)
        
        return history, loss, acc, y_preds

class ConvDenoiser_SVHN():
    def __init__(self, X, X_test, y, y_test):
        self.X = X
        self.X_test = X_test
        self.y = y
        self.y_test = y_test

    def model(self, X, X_test, y, y_test):
        X = np.array(X)
        X_test = np.array(X_test)
        y = np.array(y)  
        y_test = np.array(y_test)

        # reshape image arrays to match input shape of the autoencoder
        X = X.reshape((-1, 32, 32, 3))
        X_test = X_test.reshape((-1, 32, 32, 3))

        # normalize pixel values to the range [0, 1]
        X = X / 255.0
        X_test = X_test / 255.0

        # network params
        input_shape = (32, 32, 3)
        batch_size = 64
        validation_split = 0.25
        epochs = 25
        kernel_size = 6
        latent_dim = 64
        layer_filters = [64, 256]

        # autoencoder
        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs

        # Stack of Conv2D(64)-Conv2D(128)
        for filters in layer_filters:
            x = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=2,
                       activation='relu',
                       padding='same')(x)

        shape = K.int_shape(x)

        # latent vector
        x = Flatten()(x)
        latent = Dense(latent_dim, name='latent_vector')(x)

        # encoder
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # decoder
        latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
        x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)
        
        # stack of Conv2DTranspose(64)-Conv2DTranspose(32)
        for filters in layer_filters[::-1]: # last layer
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=2,
                                activation='relu',
                                padding='same')(x)
            
        x = Flatten()(x)

        outputs = Dense(10, activation='sigmoid', name='decoder_output')(x)

        # instantiate decoder
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # instantiate autoencoder
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        autoencoder.summary()

        # train
        history = autoencoder.fit(X_test, y_test,
                                validation_data=[X, y],
                                # validation_split=validation_split,
                                epochs=epochs,
                                batch_size=batch_size)

        # score
        loss, acc = autoencoder.evaluate(X, y, verbose=0)

        autoencoder.save('denoising_autoenconder.model')

        # predict autoencoder output
        predictions = autoencoder.predict(X)

        return history, loss, acc, predictions