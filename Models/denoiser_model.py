from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
import numpy as np

# Define the denoiser architecture
class ConvDenoiser():
    def __init__(self, X_noisy, X_clean, y_noisy, y_clean):
        self.X_noisy = X_noisy
        self.X_clean = X_clean
        self.y_noisy = y_noisy
        self.y_clean = y_clean

    def model(self, X_noisy, X_clean, y_noisy, y_clean):
        # Convert image lists to NumPy arrays
        X_noisy = np.array(X_noisy)
        X_clean = np.array(X_clean)
        y_noisy = np.array(y_noisy)
        y_clean = np.array(y_clean)

        # Reshape image arrays to match the input shape of the autoencoder
        X_noisy = X_noisy.reshape((-1, 28, 28, 1))
        X_clean = X_clean.reshape((-1, 28, 28, 1))
        # clean_train = clean_train.reshape((-1, 28, 28, 1))

        # Normalize pixel values to the range [0, 1]
        X_noisy = X_noisy / 255.0
        X_clean = X_clean / 255.0
        # clean_train = clean_train / 255.0

        # Network parameters
        input_shape = (28, 28, 1)
        batch_size = 128
        kernel_size = 6
        latent_dim = 64
        layer_filters = [64, 256]

        # Build the autoencoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs

        # Stack of Conv2D(64)-Conv2D(128)
        for filters in layer_filters:
            x = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=2,
                       activation='relu',
                       padding='same')(x)

        # Shape info needed to build decoder model
        shape = K.int_shape(x)

        # Generate the latent vector
        x = Flatten()(x)
        latent = Dense(latent_dim, name='latent_vector')(x)

        # encoder model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()

        # decoder model
        latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
        x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        # Stack of Conv2DTranspose(64)-Conv2DTranspose(32)
        for filters in layer_filters[::-1]:
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=2,
                                activation='relu',
                                padding='same')(x)

        # Reconstruct the denoised input
        outputs = Conv2DTranspose(filters=1,
                                  kernel_size=kernel_size,
                                  padding='same',
                                  activation='sigmoid',
                                  name='decoder_output')(x)

        # Instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # Instantiate autoencoder model
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        autoencoder.summary()

        # Compile the autoencoder model
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        autoencoder.summary()

        # Train the autoencoder
        history = autoencoder.fit(X_noisy,
                                X_clean,
                                validation_split=0.25,
                                epochs=100,
                                batch_size=batch_size)

        # score
        score = autoencoder.evaluate(X_noisy, X_clean)

        autoencoder.save('denoising_autoenconder.model')

        # Predict the autoencoder output from corrupted test images
        predictions = autoencoder.predict(X_noisy)

        return history, score, predictions
    
 class CNN():
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

        X = X.reshape((-1, 32, 32, 3))
        X_test = X_test.reshape((-1, 32, 32, 3))

         # normalize pixel values to the range [0, 1]
        X = X / 255.0
        X_test = X_test / 255.0

        # network params
        input_shape = (32, 32, 3)
        batch_size = 64
        validation_split = 0.25
        epochs = 100
        kernel_size = 3
        latent_dim = 64
        layer_filters = [16, 32]

        # encoder
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

        # dtack of Conv2DTranspose(64)-Conv2DTranspose(32)
        for filters in layer_filters[::-1]:
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=2,
                                activation='relu',
                                padding='same')(x)

        outputs = Conv2DTranspose(filters=3,
                                  kernel_size=kernel_size,
                                  padding='same',
                                  activation='sigmoid',
                                  name='decoder_output')(x)

        # instantiate decoder
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        # instantiate autoencoder
        model = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        model.summary()

        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.summary()

        # train model
        history = model.fit(X,
                    X_test,
                    validation_split=validation_split,
                    epochs=epochs,
                    batch_size=batch_size)
        
        # score
        score = model.evaluate(X, X_test)

        model.save('cnn_model.h5')

        # predict autoencoder output
        predictions = model.predict(X_test)

        return history, score, predictions

