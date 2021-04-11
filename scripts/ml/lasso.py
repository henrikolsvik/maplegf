import sys
from sklearn import linear_model
from ml_interface import Mlinterface
import sklearn.linear_model


class Lasso(Mlinterface):

    def machine_learning_service(self, input_samples_file, input_samples_parameters_file, input_target_file,
                                 output_filename, config_file):
        read_samples_with_names, target = self.load_files(input_samples_file, input_target_file)
        self.read_config(config_file)

        target = self.targets_to_int(target)
        train_sample, train_target, test_sample, test_target, test_name = \
            self.n_split_shuffle(read_samples_with_names, target, int(self.config["n"]))

        if self.config["positive"].lower() == "true":
            clf = linear_model.Lasso(positive=True, tol=float(self.config["tol"]))
        else:
            clf = linear_model.Lasso(positive=False, tol=float(self.config["tol"]))
        score, predictions = self.make_predictions(clf, train_sample, train_target, test_sample, test_target, test_name)

        auc_scores = []
        for i in range(0, len(predictions)):
            auc_scores.append(sklearn.metrics.roc_auc_score(predictions[i][2], predictions[i][1]))

        self.write_results(output_filename, input_samples_file, input_samples_parameters_file, auc_scores, target)


if __name__ == '__main__':
    lasso = Lasso()
    lasso.machine_learning_service(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
