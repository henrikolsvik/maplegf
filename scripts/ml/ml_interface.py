import sklearn
import time
import numpy as np


class Mlinterface:

    def target_strings_to_int(self, target):
        target_int = []

        target_set = set(target)
        if len(target_set) == 2:
            for i in range(0, len(target)):
                if target[i] == "M":
                    target[i] = 1
                else:
                    target[i] = "0"

        for item in target:
            target_int.append(int(item))

        return target_int

    def write_results(self, output_filename, score, predictions):
        file = open(output_filename, "w")
        file.write(Mlinterface.generate_result_text(self, score, predictions))
        file.close()

    def generate_result_text(self, score, predictions):
        results_string = "Total Accuracy Score Of: " + "{:.2f}".format(np.array(score).sum() / len(score) * 100) \
                         + "%.\n" + "Results of individual runs: " + str(score) + "\n**** \n\n"

        return results_string

    def load_files(self, input_samples_file, input_target_file):
        return Mlinterface.read_sample_file(self, input_samples_file), Mlinterface.read_target_file(self,
                                                                                                    input_target_file)

    def read_sample_file(self, filename):
        file = open(filename, "r")
        lines = file.readlines()
        data = []
        sample_names = []
        for i in range(1, len(lines)):
            line = lines[i].replace("\n", "")
            line = line.split(",")
            sample_names.append(line.pop(0))
            line_float = []
            for element in line:
                line_float.append(float(element))
            data.append(line_float)
        file.close()

        return [data, sample_names]

    def read_target_file(self, filename):
        file = open(filename, "r")
        lines = file.readlines()
        data = []
        for i in range(0, len(lines)):
            line = lines[i].replace("\n", "")
            line = line.split(",")
            data.append(line[1])
        file.close()

        return data

    def make_predictions(self, clf, train_sample, train_target, test_sample, test_target, test_name):
        predictions, score = [], []
        for i in range(0, len(train_sample)):
            clf.fit(np.array(train_sample[i]), np.array(train_target[i]))
            score.append(clf.score(np.array(test_sample[i]), np.array(test_target[i])))
            predictions.append([test_name[i], clf.predict(np.array(test_sample[i])), test_target[i]])

        return score, predictions

    def make_predictions_plc(self, clf, train_sample, train_target, test_sample, test_target, test_name):
        predictions, score = [], []
        for i in range(0, len(train_sample)):
            clf.fit(train_sample[i], train_target[i])
            score.append(clf.score(test_sample[i], test_target[i]))
            predictions.append([test_name[i], clf.predict(test_sample[i]), test_target[i]])

        return score, predictions

    def n_split_shuffle(self, samples, target, n):
        bound_samples_and_targets, train_sample, test_sample, test_target, train_target, test_name = [], [], [], \
                                                                                                     [], [], []
        for i in range(0, len(samples[1])):
            bound_samples_and_targets.append([samples[1][i], samples[0][i], target[i]])

        kf = sklearn.model_selection.KFold(n, shuffle=True, random_state=int(time.time())) \
            .split(bound_samples_and_targets)

        for items in kf:
            train_add_sample, train_add_target, test_add_sample, test_add_target, test_add_name = [], [], [], [], []

            for item in items[0]:
                train_add_sample.append(bound_samples_and_targets[item][1])
                train_add_target.append(bound_samples_and_targets[item][2])
            for item in items[1]:
                test_add_name.append(bound_samples_and_targets[item][0])
                test_add_sample.append(bound_samples_and_targets[item][1])
                test_add_target.append(bound_samples_and_targets[item][2])

            train_sample.append(train_add_sample)
            train_target.append(train_add_target)
            test_sample.append(test_add_sample)
            test_target.append(test_add_target)
            test_name.append(test_add_name)

        return train_sample, train_target, test_sample, test_target, test_name
