from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class EmotionDetectionModel:
    """
        Sequential Deep Convolutional Neural Network Model in Keras for detecting emotions
    """

    def __init__(self, data: dict, num_of_emotions: int, image_input_size=(48, 48), model_path="", batch_size=32, epochs=50):
        """
        :param data: a dictionary with different processed and formatted arrays of data
        :param num_of_emotions: number of emotions in the model
        :param image_input_size: size of the input images
        :param model_path: path where the model will be stored
        :param batch_size: batch size used for training
        :param epochs: epochs used for training
        """
        self.data = data
        self.num_of_emotions = num_of_emotions
        self.image_input_size = image_input_size
        self.model_path = model_path
        self.batch_size = batch_size
        self.epochs = epochs

    def create_sequential_model(self):
        """
        Creates a sequential model for the deep convolutional neural network
        :return: a sequential model object
        """
        print("Creating a Sequential model")

        # create a sequential model
        model_int = models.Sequential()

        # encoding block 1
        model_int.add(Conv2D(32, (3, 3), input_shape=(self.image_input_size[0], self.image_input_size[1], 1)))
        model_int.add(Activation("relu"))
        model_int.add(MaxPool2D((2, 2)))
        model_int.add(BatchNormalization())

        # encoding block 2
        model_int.add(Conv2D(64, (3, 3)))
        model_int.add(Activation("relu"))
        model_int.add(MaxPool2D((2, 2)))

        # encoding block 3
        model_int.add(Conv2D(64, (3, 3)))
        model_int.add(Dropout(0.1))
        model_int.add(Activation("relu"))
        model_int.add(MaxPool2D((2, 2)))
        model_int.add(BatchNormalization())

        # encoding block 4
        model_int.add(Conv2D(128, (3, 3)))
        model_int.add(Activation("relu"))
        model_int.add(MaxPool2D((2, 2)))

        # decoding block 1
        model_int.add(Flatten())
        model_int.add(Dense(128))
        model_int.add(Activation("relu"))

        # decoding block 2
        model_int.add(Flatten())
        model_int.add(Dense(64))
        model_int.add(Activation("relu"))

        # decoding block 3
        model_int.add(Flatten())
        model_int.add(Dense(32))
        model_int.add(Activation("relu"))

        # decoding block 4
        model_int.add(Dense(self.num_of_emotions))
        model_int.add(Activation('softmax'))

        # compile the model using categorical cross entropy as the loss function and Adam optimization as the optimizer
        model_int.compile(optimizer="adam",
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

        print(model_int.summary())
        print("Model Creation Complete!")
        return model_int

    def train_model(self):
        """
        This function is used to train the CNN sequential model
        :return: history object returned by Keras's model.fit() function
        """

        # here are three callback functions to aid the training, they are called after every epoch

        # saves the best version of the model in a file and prints some stats about the epoch
        checkpoint = ModelCheckpoint(self.model_path,
                                     monitor='val_loss',
                                     mode='min',
                                     save_best_only=True,
                                     verbose=1)

        # stops the training early if the learning starts to become stagnant for 5 continuous epochs
        earlystop = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=5,
                                  verbose=1,
                                  restore_best_weights=True
                                  )

        # reduces the learning rate as the validation loss starts to stagnate a little for at least 3 continuous epochs
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.2,
                                      patience=3,
                                      verbose=1,
                                      min_delta=0.0001)

        training_model = self.create_sequential_model()

        # start training the model
        history = training_model.fit(
            self.data["train_data_augmented"],
            validation_data=(self.data["val_data"], self.data["val_labels"]),
            epochs=self.epochs,
            callbacks=[earlystop, checkpoint, reduce_lr],
            batch_size=self.batch_size,
            class_weight=self.data["weights"]
        )

        # Evaluate the validation accuracy and loss
        val_loss, val_acc = training_model.evaluate(self.data["test_data"], self.data["test_labels"])
        print(f"Validation Accuracy: {val_acc},\nValidation Loss: {val_loss}")

        return history
