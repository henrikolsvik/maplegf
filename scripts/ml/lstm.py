import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.models import Sequential
import numpy as np
from tensorflow.keras import regularizers

from ml_interface import Mlinterface


class LSTM(Mlinterface):

    def machine_learning_service(self, input_samples_file, input_samples_parameters_file, input_target_file,
                                 output_filename, config_file):
        read_samples_with_names, target = self.load_files(input_samples_file, input_target_file)
        target = self.targets_to_int(target)
        self.read_config(config_file)

        train_sample, train_target, test_sample, test_target, test_name = \
            self.n_split_shuffle(read_samples_with_names, target, int(self.config["n"]))

        max_value = 0
        for set in train_sample:
            for item in set:
                if max(item) > max_value:
                    max_value = int(max(item))+1
        max_value = max_value + int(self.config["lstm_normalization_multiplier"])

        score = []

        for i in range(0, len(train_sample)):
            iter_train_sample = [train_sample[i]]
            iter_train_sample = np.array(iter_train_sample)
            iter_train_sample = iter_train_sample.reshape(len(iter_train_sample[0]), 1, len(iter_train_sample[0][0])).tolist()
            iter_train_target = [int(k) for k in train_target[i]]

            model = Sequential()
            #model.add(layers.Embedding(max_value, int(self.config["embedding"]), input_shape=(len(train_sample[0][0]), )))#, input_shape=(len(train_sample[0][0]), ))) #Max value, 32,
            model.add(layers.LSTM(int(self.config["LSTM_depth"])))


            model.add(layers.Dense(2, activation='softmax'))



            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            model.fit(iter_train_sample, iter_train_target, batch_size=int(self.config["batch_size"]),
                      epochs=int(self.config["epochs"]))

            keras.backend.clear_session()

            iter_test_sample = [test_sample[i]]
            iter_test_sample = np.array(iter_test_sample)
            iter_test_sample = iter_test_sample.reshape(len(iter_test_sample[0]), 1, len(iter_test_sample[0][0])).tolist()
            iter_test_target = [int(k) for k in test_target[i]]

            #model.fit(iter_train_sample, iter_train_target, batch_size=int(self.config["batch_size"]), epochs=int(self.config["epochs"]))
            score.append(model.evaluate(iter_test_sample, iter_test_target, verbose=2)[1])

        self.write_results(output_filename, input_samples_file, input_samples_parameters_file, score, target)


if __name__ == '__main__':
    rf = LSTM()
    rf.machine_learning_service(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
