from demo import Demo
from data import DataManager
from model import EmotionDetectionModel
from plots import ModelPlots


# all the emotions and their labels that are available in the dataset
ALL_EMOTIONS = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# emotions that we have selected for training
EMOTIONS = {3: 'Happy', 6: 'Neutral'}

# recursive path where all the output files would exist
OUT_PATH = '../out/**/*'

# path to the dataset
DATASET_PATH = '../data/fer2013.csv'

# path to our model
MODEL_PATH = '../out/emotion_model.h5'

# input size of the images within the dataset
INPUT_IMAGE_SIZE = (48, 48)

# number of emotions selected for training
NUM_OF_EMOTIONS = len(EMOTIONS.keys())

# the batch size used for training
BATCH_SIZE = 32

# the epochs used for training
EPOCHS = 50

# boolean variable to decide if demo should be run
RUN_DEMO = True

# boolean variable to decide if model should be trained
TRAIN_MODEL = True

# boolean variable to decide if things should be imported from scratch, keep True for good results
IMPORT_FROM_SCRATCH = True

# boolean variables to decide if the plots and graphs should
MAKE_PLOTS = True


if __name__ == '__main__':

    data_mng = DataManager(
        path=DATASET_PATH,
        emotions=EMOTIONS,
        image_size=INPUT_IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        import_from_scratch=IMPORT_FROM_SCRATCH
    )
    data = data_mng.get_formatted_data()

    if TRAIN_MODEL:
        model = EmotionDetectionModel(
            data=data,
            num_of_emotions=NUM_OF_EMOTIONS,
            image_input_size=INPUT_IMAGE_SIZE,
            model_path=MODEL_PATH,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS
        ).train_model()

    if RUN_DEMO:
        mapped_emotions = data["mapped_emotions"]
        demo = Demo(
            emotions=mapped_emotions,
            image_input_size=INPUT_IMAGE_SIZE,
            model_path=MODEL_PATH
        ).run_demo()

    if MAKE_PLOTS:
        ModelPlots(
            path=OUT_PATH,
            data_manager=data_mng
        ).make_plots()