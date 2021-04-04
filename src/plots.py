import matplotlib.pyplot as plt
import numpy as np
import glob
from tensorflow.keras.models import load_model
from data import DataManager


class ModelPlots:
    """
    Generates graphs and plots for the results and data used by the model
    """

    def __init__(self, path: str, data_manager: DataManager):
        """
        :param path: path to the output directory
        :param data_manager: data manager object
        """

        self.path = path
        self.data_manager = data_manager

        # mapped emotions from the data
        self.mapped_emotions = data_manager.get_mapped_emotions()

        # list of all the emotions to get the data for
        self.emotions_to_show = list(self.mapped_emotions.keys())

    def make_plot_for_file(self, path: str):
        """
        Creates a new plot comparing the Loss vs Validation Loss and the Accuracy vs Validation Accuracy of our model
        :param path: string path to the output file
        """

        out = self.data_manager.read_output_file(path)

        fig = plt.figure(figsize=(15, 9))
        fig.tight_layout()
        fig.canvas.manager.set_window_title("Accuracy vs Loss for: " + out["model"])

        plt.subplot(1, 2, 1)
        plt.title("Loss vs Validation Loss")
        plt.plot(out["labels_epoch"], out["val_loss"], '-b', label="val_loss")
        plt.plot(out["labels_epoch"], out["loss"], '--k', label="loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yticks(np.arange(0, 1.6, step=0.1))
        plt.xticks(np.arange(0, len(out["labels_epoch"]) + 1, step=3))
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title("Accuracy vs Validation Accuracy")
        plt.plot(out["labels_epoch"], out["val_acc"], '-r', label="val_acc")
        plt.plot(out["labels_epoch"], out["acc"], '--g', label="acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xticks(np.arange(0, len(out["labels_epoch"]) + 1, step=3))
        plt.legend()

    @staticmethod
    def plot_images_on_figure(fig, columns, rows, images, labels):
        """
        Plots the given images onto the figure using the number of rows and columns as guides

        :param fig: Figure object from PyPlot
        :param columns: number of columns
        :param rows: number of rows
        :param images: flattened array of images
        :param labels: flattened array of labels corresponding to the images
        """

        fig.tight_layout()
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            plt.title(str(i) + ": " + labels[i - 1])
            plt.gray()
            plt.imshow(images[i-1])

    def show_sample_dataset(self, number_of_samples=5):
        """
        Shows some randomly selected images from the dataset for the given emotions on a new figure
        :param number_of_samples: number of sample images to select
        """

        labels = []
        for i in self.emotions_to_show:
            for j in range(number_of_samples):
                labels.append(self.mapped_emotions[i])

        images = self.data_manager.get_images_from_data(number_of_images=number_of_samples, data_type="Training")

        fig = plt.figure(figsize=(15, 9))
        fig.canvas.manager.set_window_title("Training Sample From Each Category")
        self.plot_images_on_figure(fig, number_of_samples, len(self.emotions_to_show), images, labels)

    def show_class_weight(self, data_type='Training'):
        """
        Creates a new pie plot showing the distribution of the multiple categories of emotions
        :param data_type: string for the type of data needed - Training | PrivateTest | PublicTest
        """

        keys, values, totals = self.data_manager.get_weights(data_type=data_type)

        fig = plt.figure()
        fig.canvas.manager.set_window_title("Distribution of Sample Data for: " + data_type)

        plt.pie(values, labels=keys, autopct='%1.1f%%', startangle=90, colors=['#0091ea', '#cfd8dc', '#607d8b'])
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        t = ""
        for i in range(len(keys)):
            t += '{:10s} {:s} \n'.format(keys[i] + ":", str(totals[i]))

        plt.text(0.02, 0.85, t, fontsize=12, transform=plt.gcf().transFigure)

    def show_dataset_weight(self):
        """
        Creates a new figure with a pie plot showing the distribution of the entire dataset and the different subsets
        """

        value = self.data_manager.get_weights("", emotion_row=False)
        labels = ('Training', 'PublicTest', 'PrivateTest')
        printable_labels = ('Training Set', 'Validation Set', 'Test Set')

        fig = plt.figure()
        fig.canvas.manager.set_window_title("Distribution of Dataset")

        plt.pie(value[1], labels=printable_labels, autopct='%1.1f%%', startangle=90, colors=['#0091ea', '#cfd8dc', '#607d8b'])
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        t = ""
        for i in range(len(labels)):
            keys, values, totals = self.data_manager.get_weights(labels[i], emotion_row=False)
            t += '{:20s} {:s} \n'.format(printable_labels[i] + ":", str(totals[i]))

        plt.text(0.02, 0.80, t, fontsize=12, transform=plt.gcf().transFigure)

    @staticmethod
    def show_validation_plots(path: str):
        """
        Creates a new validation accuracy vs validation loss bar graph using the output file
        :param path: path to the output file
        """

        file = open(path, "r")
        lines = file.readlines()

        filtered_lines = list(filter(lambda k: 'Validation' in k, lines))
        filtered_lines = list(map(lambda k: k.replace(",", ""), filtered_lines))
        filtered_lines = list(map(lambda k: k.replace("\n", ""), filtered_lines))
        filtered_lines = np.array(list(map(lambda k: k.split(": "), filtered_lines)))

        y_pos = np.arange(2)
        performance = list(np.around(filtered_lines[:, 1].astype(float), 2))

        fig = plt.figure(figsize=(13, 5))
        fig.canvas.manager.set_window_title("Our Model's Results")
        plt.barh(y_pos, performance, align='center', alpha=0.5)
        plt.yticks(y_pos, list(filtered_lines[:, 0]))
        plt.xticks(np.arange(0, 1.2, step=0.1))
        plt.title("Validation Accuracy vs Validation Loss")

        for index, value in enumerate(performance):
            plt.text(value, index, str(value))

    def show_predictions_on_validation_data(self, path: str):
        """
        Creates a new figure with with some randomly selected images from the validation set that are run against our model
        Displays the pictures and the percentage of confidence in the model's predictions

        :param path: path to the model file
        """

        classifier = load_model(path)
        images = self.data_manager.get_images_from_data(number_of_images=5, data_type="PublicTest")
        labels = []

        for img in images:
            preds = classifier.predict(np.expand_dims(img, axis=0))[0]
            pred_index = preds.argmax()
            label = self.mapped_emotions[self.emotions_to_show[pred_index]] + " %.2f%%" % float(preds[pred_index] * 100)
            labels.append(label)

        fig = plt.figure(figsize=(15, 9))
        fig.canvas.manager.set_window_title("Predictions Using Our Model")
        self.plot_images_on_figure(fig, 5, len(self.emotions_to_show), images, labels)

    @staticmethod
    def show_total_time(path: str):
        """
        Prints the total amount of time taken to train the entire model from the output file
        :param path: path to the output file
        :return:
        """
        file = open(path, "r")
        lines = file.readlines()

        filtered_lines = filter(lambda k: '[==============================]' in k, lines)
        filtered_lines = map(lambda k: k.split(" "), filtered_lines)
        filtered_lines = map(lambda k: k[3], filtered_lines)
        filtered_lines = map(lambda k: k.replace("s", ""), filtered_lines)
        filtered_lines = list(map(lambda k: int(k), filtered_lines))
        total_time_taken = sum(filtered_lines)/60
        print("Total time taken to train our model is {:f} minutes".format(total_time_taken))

    def make_plots(self):
        """
        Creates various different plots visualizing the data from the plot
        """

        for f_path in glob.iglob(self.path, recursive=True):
            if f_path.endswith(".txt"):
                self.show_total_time(f_path)
                self.show_validation_plots(f_path)
                self.make_plot_for_file(f_path)
            if f_path.endswith(".h5"):
                self.show_predictions_on_validation_data(f_path)

        self.show_sample_dataset(number_of_samples=5)
        self.show_class_weight('Training')
        self.show_class_weight('PrivateTest')
        self.show_class_weight('PublicTest')
        self.show_dataset_weight()
        plt.show()
