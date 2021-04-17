import sys

from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
import numpy as np

from ml_interface import Mlinterface


class LSTM(Mlinterface):

    def machine_learning_service(self, input_samples_file, input_samples_parameters_file, input_target_file,
                                 output_filename, config_file):
        read_samples_with_names, target = self.load_files(input_samples_file, input_target_file)
        target = self.targets_to_int(target)
        self.read_config(config_file)

        train_sample, train_target, test_sample, test_target, test_name = \
            self.n_split_shuffle(read_samples_with_names, target, int(self.config["n"]))

        score = []

        for i in range(0, len(train_sample)):
            iter_train_sample = [train_sample[i]]
            iter_train_sample = np.array(iter_train_sample)
            iter_train_sample = iter_train_sample.reshape(len(iter_train_sample[0]), 1, len(iter_train_sample[0][0])).tolist()
            iter_train_target = [int(k) for k in train_target[i]]

            model = Sequential()
            for q in range(1, int(self.config["num_layers"])):
                model.add(layers.LSTM(int(self.config["LSTM_depth"]), return_sequences=True))
                model.add(Dropout(0.2))
            model.add(layers.LSTM(int(self.config["LSTM_depth"])))
            model.add(layers.Dense(int(self.config["denselayer_size"]), activation='softmax'))

            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            model.fit(iter_train_sample, iter_train_target, batch_size=int(self.config["batch_size"]),
                      epochs=int(self.config["epochs"]))

            iter_test_sample = [test_sample[i]]
            iter_test_sample = np.array(iter_test_sample)
            iter_test_sample = iter_test_sample.reshape(len(iter_test_sample[0]), 1, len(iter_test_sample[0][0])).tolist()
            iter_test_target = [int(k) for k in test_target[i]]

            score.append(model.evaluate(iter_test_sample, iter_test_target, verbose=2)[1])

        self.write_results(output_filename, input_samples_file, input_samples_parameters_file, score, target)


if __name__ == '__main__':
    rf = LSTM()
    rf.machine_learning_service(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
