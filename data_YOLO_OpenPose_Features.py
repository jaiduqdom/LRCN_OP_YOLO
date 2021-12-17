"""
Clase para gestionar el generador de datos de OpenPose, YOLO e Inception features
"""
import csv
import numpy as np
import random
import os.path
import threading
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class DataSet():

    def __init__(self, seq_length=90, class_limit=None):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        #self.sequence_path = os.path.join('/datos','STAIR_Train_Test', 'sequences_features_OpenPose_YOLO')
        self.sequence_path = os.path.join('/datos','NTURGBD', 'sequences_features_OpenPose_YOLO')
        self.max_frames = 30000  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning.
        self.data = self.clean_data()
        # print(self.data)

        # Longitud del dataset de entrenamiento/validation y test
        # self.len_Training = 0
        # self.len_Validation = 0
        # self.len_Test = 0
        train, validation, test = self.split_train_validation_test()
        self.len_Training = len(train)
        self.len_Validation = len(validation)
        self.len_Test = len(test)

    @staticmethod
    def get_data():
        """Load our data from file."""
        #with open(os.path.join('/datos','STAIR_Train_Test','data_file_filtrado_shuffle.csv'), 'r') as fin:
        with open(os.path.join('/datos','NTURGBD','data_file_filtrado_shuffle.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data

    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean = []
        for item in self.data:
            if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
                    and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert len(label_hot) == len(self.classes)

        return label_hot

    def split_train_validation_test(self):
        """Split the data into train/validation and test groups."""
        train = []
        test = []
        validation = []
        
        # Repartimos un 20% a validación
        ficheros = []
        acciones = []
        for item in self.data:
            if item[0] == 'train':        
                ficheros.append(item[2])
                acciones.append(item[1])
            else:
                test.append(item)
        acciones = np.array(acciones)
        ficheros = np.array(ficheros)
        f_train, f_val, a_train, a_val = train_test_split(ficheros, acciones, test_size=0.20, random_state=42, stratify=acciones)
        f_sort = f_val.tolist()
        f_sort.sort()

        for item in self.data:
            if item[2] in f_sort:
                validation.append(item)
            else:
                train.append(item)

        return train, validation, test

    @threadsafe_generator
    def frame_generator(self, batch_size, train_validation_test):
        """Return a generator that we can use to train on."""

        # Get the right dataset for the generator.
        train, validation, test = self.split_train_validation_test()
        data = None
        if train_validation_test == 'train':
            data = train
        if train_validation_test == 'validation':
            data = validation
        if train_validation_test == 'test':
            data = test

        print("Creating %s generator with %d samples." % (train_validation_test, len(data)))

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                matriz_YOLO = None
                matriz_OpenPose = None
                matriz_Features = None

                # Get a random sample.
                sample = random.choice(data)

                # Get the sequence from disk.
                matriz_YOLO, matriz_OpenPose, matriz_Features = self.get_extracted_YOLO_OP_Feature_sequence(sample)

                if matriz_YOLO is None or matriz_OpenPose is None or matriz_Features is None:
                    print("Archivo: " + str(sample))
                    raise ValueError(str(sample) + "Can't find sequence. Did you generate them?")

                # Para poderlo utilizar con Tensorflow facilmente, hacemos un flatten a las 3 matrices y las
                # fusionamos. De esta manera recibiremos un unico tensor y luego lo desacoplaremos dentro
                # de nuestro modelo
                m1 = matriz_YOLO.flatten()
                m2 = matriz_OpenPose.flatten()
                m3 = matriz_Features.flatten()
                matrizTotal = np.concatenate((m1, m2, m3))
                
                # print(str(len(matrizTotal)))
                # print(str(len(m1)))
                # print(str(len(m2)))
                # print(str(len(m3)))                

                X.append(matrizTotal)
                # print(str(len(np.array(X[0]))))
                y.append(self.get_class_one_hot(sample[1]))

            yield np.array(X), np.array(y)

    def get_extracted_YOLO_OP_Feature_sequence(self, sample):
        """Get the saved extracted features."""
        filename = sample[2]
        path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) + \
            '-yolo_op_features.npz')
        if os.path.isfile(path):
            npzfile = np.load(path)
            matriz_YOLO = npzfile['YOLO']
            matriz_OpenPose = npzfile['OpenPose']
            matriz_Features = npzfile['FeaturesInception']
            return matriz_YOLO, matriz_OpenPose, matriz_Features
        else:
            return None, None, None
