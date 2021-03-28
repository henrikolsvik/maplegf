from sklearn import svm
import sys
from ml_interface import Mlinterface


class SvmClassifier(Mlinterface):

    def machine_learning_service(self, input_samples_file, input_samples_parameters_file, input_target_file,
                                 output_filename, config_file):
        read_samples_with_names, target = self.load_files(input_samples_file, input_target_file)
        self.read_config(config_file)

        if self.config["ufs_stage"] == "pre":
            samples_with_names = self.do_usf(read_samples_with_names, target)
        else:
            samples_with_names = read_samples_with_names

        train_sample, train_target, test_sample, test_target, test_name = \
            self.n_split_shuffle(samples_with_names, target, int(self.config["n"]))

        clf = svm.SVC(C=float(self.config["C"]))
        score, predictions = self.make_predictions(clf, train_sample, train_target, test_sample, test_target, test_name)

        self.write_results(output_filename, input_samples_file, input_samples_parameters_file, score, target)


if __name__ == '__main__':
    svm_classifier = SvmClassifier()
    svm_classifier.machine_learning_service(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
