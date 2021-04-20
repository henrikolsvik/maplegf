from sklearn.neural_network import MLPClassifier
from lime import lime_tabular
import numpy as np
import sys
from ml_interface import Mlinterface


class MLPNN(Mlinterface):

    def machine_learning_service(self, input_samples_file, input_samples_parameters_file, input_target_file,
                                 output_filename, config_file):
        feature_names = self.get_feature_names(input_samples_file)
        read_samples_with_names, target = self.load_files(input_samples_file, input_target_file)
        self.read_config(config_file)

        train_sample, train_target, test_sample, test_target, test_name = \
            self.n_split_shuffle(read_samples_with_names, target, int(self.config["n"]))

        clf = MLPClassifier(max_iter=int(self.config["max_iter"]),
                            hidden_layer_sizes=int(self.config["hidden_layer_sizes"]),
                            alpha=float(self.config["alpha"]))
        score, predictions = self.make_predictions(clf, train_sample, train_target, test_sample, test_target, test_name)

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(train_sample[0]),
            training_labels=np.array(train_target[0]),
            feature_names=feature_names,
            class_names=clf.classes_
        )

        exp = explainer.explain_instance(
            data_row=np.array(test_sample[0][0]),
            predict_fn=clf.predict_proba,
            num_features=50
        )

        file = open("results/mlpnn_explain.html", "w")
        file.write("Trying to explain " + str(test_name[0][0]) + ". Is " + str(test_target[0][0]) + "</br>")
        file.write("Predicted as " + str(predictions[0][1][0]) + "</br>")
        file.write(exp.as_html())
        file.close()
        self.write_results(output_filename, input_samples_file, input_samples_parameters_file, score, target)


if __name__ == '__main__':
    mlpnn = MLPNN()
    mlpnn.machine_learning_service(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
