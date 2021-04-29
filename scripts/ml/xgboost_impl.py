import sys
from ml_interface import Mlinterface
import xgboost as xgb


class XGBooster(Mlinterface):

    def machine_learning_service(self, input_samples_file, input_samples_parameters_file, input_target_file,
                                 output_filename, config_file):
        feature_names = self.get_feature_names(input_samples_file)
        read_samples_with_names, target = self.load_files(input_samples_file, input_target_file)
        self.read_config(config_file)

        train_sample, train_target, test_sample, test_target, test_name = \
            self.n_split_shuffle(read_samples_with_names, target, int(self.config["n"]))

        xgb_model = xgb.XGBClassifier(max_depth=int(self.config["max_depth"]))
        score, predictions = self.make_predictions(xgb_model, train_sample, train_target, test_sample, test_target,
                                                   test_name)

        exp, combined_results = self.explain_results(train_sample, train_target, feature_names, xgb_model, test_sample)
        self.write_explanation(exp, combined_results, test_name, test_target, predictions)
        self.write_results(output_filename, input_samples_file, input_samples_parameters_file, score, target)


if __name__ == '__main__':
    xgbooster = XGBooster()
    xgbooster.machine_learning_service(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
