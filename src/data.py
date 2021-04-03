import numpy as np
import pandas as pd
import random

from tensorflow.keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator


class DataManager:
    """
        Prepares the dataset for the various uses
    """

    def __init__(self, path: str, emotions: dict, image_size=(48, 48), batch_size=32, import_from_scratch=True):
        """
        :param path: file path to the dataset/CSV file
        :param emotions: dict of emotion index and their labels
        :param image_size: image size of each input image
        :param batch_size: batch size used for the CNN model
        :param import_from_scratch: decides if we should reimport everything from CSV file
        """

        self.path = path
        self.emotions = emotions
        self.image_size = image_size
        self.batch_size = batch_size
        self.import_from_scratch = import_from_scratch

        self.data, self.mapped_emotions = self.filter_data_object()

    def get_mapped_emotions(self):
        """
        :return: Returns a dictionary of the mapped emotions and their labels
        """
        return self.mapped_emotions

    def get_image_labels(self, usage: str):
        """
        Returns all the labels for all the corresponding images, maintaining order from the dataset
        :param usage: the usage to extract - possible values for given dataset are Training, PrivateTest, and PublicTest
        :return: a numpy array of all the labels
        """

        return to_categorical(np.array(list(self.data[self.data.Usage == usage].emotion)))

    def get_image_data(self, usage: str):
        """
        Returns all the normalized image data as numpy arrays
        :param usage: the usage to extract - possible values for given dataset are Training, PrivateTest, and PublicTest
        :return: a numpy array of all the images as numpy array
        """

        # filter data based on usage
        filtered_data = self.data[self.data.Usage == usage]
        image_array = np.zeros(shape=(len(filtered_data), self.image_size[0], self.image_size[1], 1))

        # read images from the data and add them into the image array as numpy arrays
        for i, row in enumerate(filtered_data.index):
            image = np.fromstring(list(filtered_data.pixels)[i], dtype=int, sep=' ')
            image = np.reshape(image, (self.image_size[0], self.image_size[1], 1)) / 255.
            image_array[i] = image

        return image_array

    def filter_data_object(self):
        """
        Reads the csv file and filters out only the emotions we need from the dataset
        :return: the Dataframe object and a dictionary with mapped emotion indexes and their labels
        """

        emotions_to_show = list(self.emotions.keys())
        emotions_to_show.sort()

        # create a new query string needed to filter out emotions
        query_str = ""
        for i in range(len(emotions_to_show)):
            query_str = query_str + " emotion == " + str(emotions_to_show[i])
            if len(emotions_to_show) - 1 != i:
                query_str = query_str + " or"

        # read the data set from the csv file and filter out emotions that we need
        data = pd.read_csv(self.path).query(query_str)

        mapped_emotions = {}

        # map the emotions to something simpler
        for j in range(len(emotions_to_show)):
            data.loc[data.emotion == emotions_to_show[j], 'emotion'] = j
            mapped_emotions[j] = self.emotions[emotions_to_show[j]]

        return data, mapped_emotions

    @staticmethod
    def all_emotions():
        """
        All emotions that are within the dataset
        :return: a dictionary with all the emotions in the data set and their corresponding labels
        """
        return {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    def get_formatted_data(self):
        """
        Initialize all the data for training, testing and validating the neural network
        Also augments the dataset to add more variance and resilience into our model
        :return: a dictionary with different processed and formatted arrays of data
        """

        print("Reading data...")
        to_return = {}
        path_prefix = "../data/"

        if self.import_from_scratch:
            # get all the labels and save them in their respective files
            training_labels_import = self.get_image_labels('Training')
            np.save(path_prefix + "training_labels.npy", training_labels_import)
            validation_labels_import = self.get_image_labels('PrivateTest')
            np.save(path_prefix + "validation_labels.npy", validation_labels_import)
            test_labels_import = self.get_image_labels('PublicTest')
            np.save(path_prefix + "test_labels.npy", test_labels_import)

            # get all image data and save them in their respective files
            training_data_import = self.get_image_data('Training')
            np.save(path_prefix + "training_data.npy", training_data_import)
            validation_data_import = self.get_image_data('PrivateTest')
            np.save(path_prefix + "validation_data.npy", validation_data_import)
            test_data_import = self.get_image_data('PublicTest')
            np.save(path_prefix + "test_data.npy", test_data_import)

        # load data from the saved files
        to_return["train_labels"] = np.load(path_prefix + "training_labels.npy")
        to_return["val_labels"] = np.load(path_prefix + "validation_labels.npy")
        to_return["test_labels"] = np.load(path_prefix + "test_labels.npy")
        to_return["train_data"] = np.load(path_prefix + "training_data.npy")
        to_return["val_data"] = np.load(path_prefix + "validation_data.npy")
        to_return["test_data"] = np.load(path_prefix + "test_data.npy")

        # add different variations to the training data for resilience
        train_datagen = ImageDataGenerator(
            rotation_range=30,
            shear_range=0.3,
            zoom_range=0.3,
            width_shift_range=0.4,
            height_shift_range=0.4,
            horizontal_flip=True,
            fill_mode='nearest')

        # covert the data into something nice that Keras can use
        to_return["train_data_augmented"] = train_datagen.flow(
            x=to_return["train_data"],
            y=to_return["train_labels"],
            batch_size=self.batch_size,
            shuffle=True)

        # calculates the weights for the different emotions
        to_return["weights"] = (self.data[self.data.Usage == 'Training'].emotion.value_counts() / self.data[self.data.Usage == 'Training'].count()[0]).to_dict()
        to_return["mapped_emotions"] = self.mapped_emotions

        print("Data Read Complete!")
        return to_return

    def get_weights(self, data_type: str, emotion_row=True):
        """
        Gets the weight values and keys for the given data type
        :param data_type: string for the type of data needed - Training | PrivateTest | PublicTest
        :param emotion_row: decides if we are filtering emotions or the entire dataset
        :return: returns the keys, values and the totals of each of the filtered categories
        """

        filtered_data = self.data
        keys = ()

        if emotion_row:
            filtered_data = filtered_data[self.data.Usage == data_type]
            weights = (filtered_data.emotion.value_counts() / filtered_data.count()[0]).to_dict()
            keys = tuple(map(lambda k: self.mapped_emotions[k], weights.keys()))
            totals = list(filtered_data.emotion.value_counts())

        else:
            weights = (filtered_data.Usage.value_counts() / filtered_data.count()[0]).to_dict()
            totals = list(filtered_data.Usage.value_counts())

        values = list(weights.values())

        return keys, values, totals

    @staticmethod
    def read_output_file(path):
        """
        Parses the output file and formats the data for manipulation
        :param path: string path to the output file
        :return: a dictionary with different processed and formatted arrays of data
        """
        file = open(path, "r")
        lines = file.readlines()

        filtered_lines = list(filter(lambda k: '[==============================]' in k, lines))
        filtered_lines = list(map(lambda k: k.split(" "), filtered_lines))

        val_loss = []
        val_acc = []
        loss = []
        acc = []
        labels_epoch = []

        epoch = 1
        for line in filtered_lines:
            if len(line) == 17:
                loss.append(float(line[7]))
                acc.append(float(line[10]))
                val_loss.append(float(line[13]))
                val_acc.append(float(line[16].strip("\n")))
                labels_epoch.append(epoch)
                epoch = epoch + 1

        labels_epoch = list(reversed(labels_epoch))
        val_loss.reverse()
        val_acc.reverse()
        loss.reverse()
        acc.reverse()

        return {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "loss": loss,
            "acc": acc,
            "labels_epoch": labels_epoch,
            "model": path.split("/")[-2]
        }

    def get_images_from_data(self, number_of_images=5, data_type="Training"):
        """
        Gets randomly selected images from the dataset for the given emotions and data type

        :param number_of_images: number of images to sample
        :param data_type: string for the type of data needed - Training | PrivateTest | PublicTest
        :return:
        """

        images = []
        for item in list(self.mapped_emotions.keys()):
            sample = random.sample(list(self.data.query("Usage == '" + data_type + "' and emotion == " + str(item)).pixels), number_of_images)

            for img in sample:
                image = np.fromstring(img, dtype=int, sep=' ')
                image = np.reshape(image, (48, 48, 1)) / 255.
                images.append(image)

        return images
