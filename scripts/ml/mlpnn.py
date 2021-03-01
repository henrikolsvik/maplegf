from sklearn.neural_network import MLPClassifier
import sys
from ml_interface import Mlinterface


class MLPNN(Mlinterface):

    def machine_learning_service(self, input_samples_file, input_target_file, output_filename, results_spreadsheet, config_file):

        samples_with_names, target = self.load_files(input_samples_file, input_target_file)
        self.read_config(config_file)

        bound_samples_and_targets = []
        for i in range(0, len(samples_with_names[0])):
            bound_samples_and_targets.append([samples_with_names[1][i], samples_with_names[0][i], target[i]])

        train_sample, train_target, test_sample, test_target, test_name = \
            self.n_split_shuffle(samples_with_names, target, int(self.config["n"]))

        clf = MLPClassifier(max_iter=int(self.config["max_iter"]),
                            hidden_layer_sizes=int(self.config["hidden_layer_sizes"]),
                            alpha=float(self.config["alpha"]))
        score, predictions = self.make_predictions(clf, train_sample, train_target, test_sample, test_target, test_name)

        self.write_results(output_filename, input_samples_file, results_spreadsheet, score, target)


if __name__ == '__main__':
    mlpnn = MLPNN()
    mlpnn.machine_learning_service(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
