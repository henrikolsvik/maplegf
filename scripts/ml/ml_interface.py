import sklearn.model_selection
import time
import numpy as np
import datetime
import os.path
from lime import lime_tabular
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.preprocessing import Normalizer

class Mlinterface:

    def __init__(self):
        self.config = None
        self.timekeeping = {"Start_time:": datetime.datetime.now()}
        self.best_index = 0
        self.num_to_explain = 0

    def count_targets(self, target):
        num_targets = {}
        for i in range(0, len(target)):
            if target[i] not in num_targets:
                num_targets[target[i]] = 1
            else:
                num_targets[target[i]] += 1

        print("Type count")
        print(num_targets)
        return num_targets

    def targets_to_int(self, target):
        num_targets = {}
        for i in range(0, len(target)):
            if target[i] not in num_targets:
                num_targets[target[i]] = len(num_targets)
            target[i] = num_targets[target[i]]

        print("Binarified target files")
        print(num_targets)
        return target

    def read_config(self, file):
        data = open(file, "r")
        settings = {}
        for line in data:
            if line[0] != "#":
                settings[line.split("=")[0]] = line.split("=")[1].replace("\n", "")
        self.config = settings

    def combine_dicts(self, combined_appended):
        combined_final = {}
        for cur_dict in combined_appended:
            for item in cur_dict:
                if item not in combined_final.keys():
                    combined_final[item] = cur_dict[item]
                else:
                    combined_final[item] += cur_dict[item]
        for item in combined_final: combined_final[item] = combined_final[item] / len(combined_appended)
        return combined_final

    def explain_multiple(self, train_sample, train_target, feature_names, clf, test_sample, best_index, n):
        if n == 1:
            exp, combined_final, combined_final_terms, combined_final_term_features = self.explain_results(train_sample, train_target, feature_names, clf, test_sample, best_index)
        else:
            combined_appended_results, combined_appended_terms, combined_appended_term_features = [], [], []
            for i in range(0, n):
                exp, combined_results, terms, term_features = self.explain_results(train_sample, train_target, feature_names, clf, test_sample, i)
                combined_appended_results.append(combined_results)
                combined_appended_terms.append(terms)
                combined_appended_term_features.append(term_features)
            combined_final = self.combine_dicts(combined_appended_results)
            combined_final_terms = self.combine_dicts(combined_appended_terms)
            combined_final_term_features = self.combine_dicts(combined_appended_term_features)
            self.write_extra_explanation_info(combined_final_term_features, combined_final_terms, combined_final, "FINAL")
        return exp, combined_final, combined_final_terms, combined_final_term_features


    def explain_results(self, train_sample, train_target, feature_names, clf, test_sample, best_index):
        if ("Cancer" in x for x in clf.classes_):
            classes = ["Control", "Cancer_or_AA"]
        else:
            classes = [0, 1]

        print("Starting explainer. Predicting for n=", best_index)
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(train_sample[best_index]),
            training_labels=np.array(train_target[best_index]),
            feature_names=feature_names,
            class_names=classes,
            discretize_continuous=True
        )

        results = []

        if int(self.config["num_samples_to_explain"]) == 0:
            num_s_exp = len(test_sample[best_index])
        else:
            num_s_exp = int(self.config["num_samples_to_explain"])

        for n in range(0, num_s_exp):
            exp = explainer.explain_instance(
                data_row=np.array(test_sample[best_index][n]),
                predict_fn=clf.predict_proba,
                num_features=int(self.config["num_features_to_list"])
            )
            results.append(exp.as_list())
            print("Predicted sample.", int(n + 1), "/", num_s_exp)

        self.best_index = best_index
        self.num_to_explain = num_s_exp - 1

        combined_results, unique_term_range_occurences, unique_term_occurences = {}, {}, {}
        for i in range(0, len(results)):
            for q in range(0, len(results[i])):

                term_feature = results[i][q][0]
                feature_value = results[i][q][1]
                if term_feature.find(":") == -1:
                    term = term_feature[term_feature.find("IPR"):term_feature.find("IPR") + 10]
                else:
                    term = term_feature[term_feature.find(":") - 2: term_feature.find(":") + 8]

                if not term_feature in combined_results.keys():
                    combined_results[term_feature] = feature_value
                    unique_term_range_occurences[term_feature] = 1
                    if not term in unique_term_occurences.keys():
                        unique_term_occurences[term] = 1
                    else:
                        unique_term_occurences[term] += 1
                else:
                    combined_results[term_feature] += feature_value
                    unique_term_range_occurences[term_feature] += 1
                    unique_term_occurences[term] += 1

        print("Prediction complete.")

        for item in combined_results: combined_results[item] = combined_results[item] / len(results)
        self.write_extra_explanation_info(unique_term_range_occurences, unique_term_occurences, combined_results, best_index)
        return exp, combined_results, unique_term_occurences, unique_term_range_occurences

    def write_extra_explanation_info(self, unique_term_range_occurences, unique_term_occurences, combined_results, i):
        print("writing explanation")
        file = open("results/" + type(self).__name__ + "_" + str(
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")) + str(i) +  "_combined_explain_term_range_occurences.csv", "w", encoding="utf-8")
        for item in unique_term_range_occurences: file.write(item + "," + str(unique_term_range_occurences[item]) + "\n")
        file.close()

        file = open("results/" + type(self).__name__ + "_" + str(
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")) + str(i) + "_combined_explain_term_occurences.csv", "w",encoding="utf-8")
        for item in unique_term_occurences: file.write(item + "," + str(unique_term_occurences[item]) + "\n")
        file.close()

        file = open("results/" + type(self).__name__ + "_" + str(
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")) + str(i) + "_combined_explain.csv", "w", encoding="utf-8")
        for item in combined_results: file.write(item + "," + str(combined_results[item]) + "\n")
        file.close()

    def write_explanation(self, exp, combined_results, test_name, test_target, predictions):
        file = open("results/" + type(self).__name__ + "_" + str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + "_explain.html", "w", encoding="utf-8")
        file.write("Trying to explain " + str(test_name[self.best_index][self.num_to_explain]) + ". Is " + str(test_target[self.best_index][self.num_to_explain]) + "</br>")
        file.write("Predicted as " + str(predictions[self.best_index][1][self.num_to_explain]) + "</br>")
        file.write(exp.as_html())
        file.close()


    def write_results(self, output_filename, input_samples, input_samples_parameter, score, target):
        self.timekeeping["End_time:"] = datetime.datetime.now()
        self.timekeeping["Total_time:"] = (self.timekeeping["End_time:"] - self.timekeeping["Start_time:"]).total_seconds()
        self.write_txt_results(output_filename, target, score)
        self.write_csv_results(input_samples, input_samples_parameter, target, score)

    def write_csv_results(self, input_samples, input_samples_parameter, target, score):
        print(type(self))
        if not os.path.isfile("results/combined_results.csv"):
            open("results/combined_results.csv", "a").write(
                "Algorithm;Runtime in Seconds;Score;Score_STD;Baseline;Start_time;End_time;Scores;Parameters;Samples_name;Preprocessing_config\n")
        open("results/combined_results.csv", "a").write(
            type(self).__name__ + ";" +
            str(self.timekeeping["Total_time:"]) + ";" +
            str("{0:.3f}".format(np.array(score).sum() / len(score))) + ";" +
            str("{0:.3f}".format(np.array(score).std())) + ";" +
            str(self.get_baseline_accuracy(target)) + ";" +
            str(self.timekeeping["Start_time:"]) + ";" +
            str(self.timekeeping["End_time:"]) + ";" +
            str(score) + ";" +
            str(self.config) + ";" +
            str(input_samples) + ";" +
            str([str(x).replace("\n", "") for x in (open(input_samples_parameter, "r").readlines())]) + "\n")

    def write_txt_results(self, output_filename, target, score):
        file = open(output_filename, "w")
        file.write("FILE: " + output_filename + "\n")
        file.write("Baseline accuracy: {0:.2f}%".format(self.get_baseline_accuracy(target)) + "\n")
        file.write(self.generate_result_text(score))
        file.write(str(self.config))
        file.close()

    def get_baseline_accuracy(self, target):
        target_counts = self.count_targets(target)
        max_count, count = 0, 0
        for item in target_counts:
            if target_counts[item] > max_count:
                max_count = target_counts[item]
            count += target_counts[item]
        return (max_count / count) * 100

    def generate_result_text(self, score):
        return "Total Accuracy Score Of: " + "{:.2f}".format(np.array(score).sum() / len(score) * 100) \
               + "%.\n" + "Results of individual runs: " + str(score) + "\n**** \nConfig:\n"

    def load_files(self, input_samples_file, input_target_file):

        return Mlinterface.read_sample_file(self, input_samples_file), Mlinterface.read_target_file(self,
                                                                                                    input_target_file)

    def get_feature_names(self, input_samples_file):
        file = open(input_samples_file, "r")
        features = file.readline().replace("\n", "").split(",")[1:]
        file.close()
        return features

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

    def do_usf(self, input_samples, target):
        if self.config["ufs_stage"] == "pre":
            samples, names = [], []
            for i in range(0, len(input_samples[0])):
                samples.append(input_samples[0][i])
                names.append(input_samples[1][i])
                target[i] = target[i]
        else:
            samples = input_samples

        if self.config["ufs_type"] == "percent":
            filtered_terms = SelectPercentile(percentile=int(self.config["ufs_number"])).fit_transform(samples,
                                                                                                       target).tolist()
        elif self.config["ufs_type"] == "count":
            filtered_terms = SelectKBest(k=int(self.config["ufs_number"])).fit_transform(samples, target).tolist()

        return [filtered_terms, names]

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

    def lstm_upscale_normalization(self, samples):
        return [[x * int(self.config["lstm_normalization_multiplier"]) for x in y] for y in samples[0]]

    def normalize_dataset(self, samples):
        if self.config["normalize_by"] == "term":
            samples[0] = (Normalizer().fit_transform(np.array(samples[0]).transpose())).transpose().tolist()
            if self.__class__.__name__ == "LSTM":
                samples[0] = self.lstm_upscale_normalization(samples)
        if self.config["normalize_by"] == "sample":
            samples[0] = Normalizer().fit_transform(samples[0]).tolist()
            if self.__class__.__name__ == "LSTM":
                samples[0] = self.lstm_upscale_normalization(samples)
        if self.config["normalize_by"] == "sample_then_term":
            samples[0] = Normalizer().fit_transform(samples[0]).tolist()
            samples[0] = (Normalizer().fit_transform(np.array(samples[0]).transpose())).transpose().tolist()
            if self.__class__.__name__ == "LSTM":
                samples[0] = self.lstm_upscale_normalization(samples)
        return samples

    def n_split_shuffle(self, samples, target, n):
        bound_samples_and_targets, train_sample, test_sample, test_target, train_target, test_name = [], [], [], [], [], []

        samples = self.normalize_dataset(samples)

        if self.config["ufs_stage"] == "pre":
            samples = self.do_usf(samples, target)

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

            if self.config["ufs_stage"] == "kfold":
                train = self.do_usf(train_add_sample, train_add_target)
                train_add_sample = train
                test = self.do_usf(test_add_sample, test_add_target)
                test_add_sample = test

            train_sample.append(train_add_sample)
            train_target.append(train_add_target)
            test_sample.append(test_add_sample)
            test_target.append(test_add_target)
            test_name.append(test_add_name)

        return train_sample, train_target, test_sample, test_target, test_name
