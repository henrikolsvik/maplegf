import sys
import tensorflow as tf
from tensorflow import keras

from ml_interface import Mlinterface


class LSTM(Mlinterface):

    def machine_learning_service(self, input_samples_file, input_target_file, output_filename, n_split):

        samples_with_names, target = self.load_files(input_samples_file, input_target_file)

        bound_samples_and_targets = []

        for i in range(0, len(samples_with_names[0])):
            bound_samples_and_targets.append([samples_with_names[1][i], samples_with_names[0][i], target[i]])

        train_sample, train_target, test_sample, test_target, test_name = \
            self.n_split_shuffle(samples_with_names, target, n_split)

        model = tf.keras.layers.LSTM()
        model.summary()

        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.RMSprop(),
            metrics=["accuracy"],
        )

        history = model.fit(train_sample, train_target, batch_size=64, epochs=2, validation_split=0.2)

        test_scores = model.evaluate(test_sample, test_target, verbose=2)
        print("Test loss:", test_scores[0])
        print("Test accuracy:", test_scores[1])

        score, predictions = self.make_predictions(model, train_sample, train_target, test_sample, test_target, test_name)

        self.write_results(output_filename, score, predictions)


if __name__ == '__main__':
    rf = LSTM()
    rf.machine_learning_service(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
