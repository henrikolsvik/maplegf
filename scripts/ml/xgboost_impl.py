import sys
from ml_interface import Mlinterface
import xgboost as xgb


class XGBooster(Mlinterface):

    def machine_learning_service(self, input_samples_file, input_target_file, output_filename, config_file):
        samples_with_names, target = self.load_files(input_samples_file, input_target_file)
        n = self.load_config(self.read_config(config_file))

        bound_samples_and_targets = []

        for i in range(0, len(samples_with_names[0])):
            bound_samples_and_targets.append([samples_with_names[1][i], samples_with_names[0][i], target[i]])

        train_sample, train_target, test_sample, test_target, test_name = \
            self.n_split_shuffle(samples_with_names, target, n)

        xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
        score, predictions = self.make_predictions(xgb_model, train_sample, train_target, test_sample, test_target,
                                                   test_name)

        self.write_results(output_filename, score, target)

    def load_config(self, config):
        n = int(config["n"])
        return n


if __name__ == '__main__':
    xgbooster = XGBooster()
    xgbooster.machine_learning_service(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
