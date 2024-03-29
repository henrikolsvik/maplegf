import sys
import os
import datetime
import numpy as np
from term_processing_utils import find_sequences_matching_metadata_and_write_matching_metadata


def run_preprocessing(sequence_dir, metadata_filepath, sample_output_filename,
                      coverage_key_stats_filename, term_count_unprocessed_filename, term_count_processed_filename,
                      parameter_output_filename, metadata_out_filename, config_file):
    process_values = {"Start_time:": datetime.datetime.now()}

    metadata = read_metadata_file(metadata_filepath)
    config = read_config(config_file)

    sequences = [[x, ""] for x in os.listdir(sequence_dir)]

    term_count_by_sample, coverage_statistics = [], []
    sequence_file_list = find_sequences_matching_metadata_and_write_matching_metadata(sequences, metadata, metadata_out_filename)

    sequence_file_list = [x[0] for x in sequence_file_list]

    process_values["Number of samples available: "] = str(len(os.listdir(sequence_dir)))
    process_values["Number of samples included: "] = str(len(sequence_file_list))
    process_values["Number of metadata items: "] = str(len(metadata))

    for sequence_filename in sequence_file_list:
        sequence_data = read_file(sequence_dir + "/" + sequence_filename)
        term_list = get_terms(config["term_type"], sequence_data[0])
        term_count_by_sample.append(count_unique_terms(term_list, config["term_type"]))
        coverage_statistics.append(get_coverage_data(term_list))

    process_values["Number of terms before filtering: "] = str(len(get_unique_terms(term_count_by_sample)))

    if sample_output_filename is not None:
        write_term_count_overview(term_count_by_sample, term_count_unprocessed_filename)

    # Method mutates term_count_by_sample
    term_count_by_sample_limited = limit_occurrence_n_in_m_share(term_count_by_sample,
                                                                 float(config["minimum_required_coverage_count"]),
                                                                 float(config[
                                                                           "minimim_share_of_samples_with_minimum_required_coverage_count"]))

    # Logging
    process_values["Number of terms after filtering: "] = str(len(get_unique_terms(term_count_by_sample_limited)))

    if sample_output_filename is not None:
        write_terms_to_file(term_count_by_sample_limited, term_count_by_sample_limited, sequence_file_list,
                            sample_output_filename)
        write_term_count_overview(term_count_by_sample_limited, term_count_processed_filename)
        write_coverage_key_stats(coverage_statistics, sequence_file_list, coverage_key_stats_filename)

        process_values["End_time:"] = datetime.datetime.now()
        elapsed_time = process_values["End_time:"] - process_values["Start_time:"]
        process_values["Total_time"] = elapsed_time.total_seconds()
        process_values["End_time:"] = str(process_values["End_time:"])
        process_values["Start_time:"] = str(process_values["Start_time:"])

        write_preprocessing_parameter_data(parameter_output_filename, sample_output_filename, sequence_file_list,
                                           len(os.listdir(sequence_dir)), process_values, config)


def write_preprocessing_parameter_data(parameter_output_filename, sample_output_filename, sequence_file_list,
                                       sequence_dir_length, process_values, config):
    file = open(parameter_output_filename, "w")
    file.write("Parameter data for: " + sample_output_filename + "\n")
    file.write(str(process_values) + "\n")
    file.write(str(config))
    file.close()


def write_coverage_key_stats(coverage_statistics, sequence_file_list, output_name):
    key_stats, sequence_stats, all_coverage = [], [], []

    for i in range(0, len(coverage_statistics)):
        sequence_stats.append([sequence_file_list[i], np.std(coverage_statistics[i]), np.mean(coverage_statistics[i])])
        all_coverage += coverage_statistics[i]

    file = open(output_name, "w")
    file.write("TERM:,Standard deviation of coverage,Mean coverage\n")

    for item in sequence_stats:
        file.write(str(item[0]) + "," + str(item[1]) + "," + str(item[2]) + "\n")
    file.close()


def get_coverage_data(term_list):
    coverage_data = []
    for item in term_list:
        coverage_data.append(item[0])
    return coverage_data


def write_term_count_overview(term_count_by_sample, filename):
    sorted_sum_dict = sum_and_sort_terms(term_count_by_sample)
    file = open(filename, "w")
    file.write("TERM:,Count:\n")
    for item in sorted_sum_dict:
        file.write(str(item) + "," + str(sorted_sum_dict[item]) + "\n")
    file.close()


def sum_and_sort_terms(term_count_by_sample):
    sum_dict = {}
    all_unique_terms = get_unique_terms(term_count_by_sample)
    for term in all_unique_terms:
        sum_dict[term] = 0

    for sample in term_count_by_sample:
        for term in sample:
            sum_dict[term] += sample[term]

    return {key: value for key, value in sorted(sum_dict.items(), key=lambda item: item[1], reverse=True)}


def get_sequence_file_list(sequence_dir, metadata, metadata_out_filename):
    file_list = []
    metadata_out = open(metadata_out_filename, "a+")
    for sequence in metadata:
        for filename in os.listdir(sequence_dir):
            if sequence[0] in filename:
                file_list.append(filename)
                metadata_out.write(str(sequence[0]) + "," + str(sequence[1]) + "\n")
    return file_list


def write_targets_to_file(metadata, sequence_file_list, target_output_filename, binary):
    file = open(target_output_filename, "w")
    for sequence in sequence_file_list:
        for item in metadata:
            if item[0] in sequence:
                if binary == "True":
                    if item[2] == "Control":
                        file.write(str(item[0]) + "," + str(0) + "\n")
                    else:
                        file.write(str(item[0]) + "," + str(1) + "\n")
                else:
                    file.write(str(item[0]) + "," + str(item[2]) + "\n")
    file.close()


def read_metadata_file(metadata_filepath):
    file = open(metadata_filepath, "r", encoding="iso-8859-1")
    lines = file.readlines()
    data = []
    for i in range(0, len(lines)):
        line = lines[i].replace("\n", "")
        data.append(line.split(","))
    file.close()
    return data


def read_files_matching_metadata(filepath, metadata_sequences):
    data = []
    for sequence in metadata_sequences:
        for filename in os.listdir(filepath):
            if sequence[0] in filename:
                data.append(read_file(filepath + "/" + filename))
    return data


def limit_occurrence_n_in_m_share(term_count_by_sample_input, threshold_abundance, threshold_share):
    term_count_by_sample = term_count_by_sample_input
    unique_terms = get_unique_terms(term_count_by_sample)
    num_of_samples = len(term_count_by_sample)

    terms_to_remove = []

    for term in unique_terms:
        term_threshold_passes = 0
        for sample in term_count_by_sample:
            if sample.get(term) is not None:

                if sample[term] >= threshold_abundance:
                    term_threshold_passes += 1
                else:
                    sample[term] = 0
            else:
                sample[term] = 0
        if term_threshold_passes / num_of_samples <= threshold_share:
            terms_to_remove.append(term)

    for term in terms_to_remove:
        for sample in term_count_by_sample:
            sample.pop(term, None)

    return term_count_by_sample


def write_terms_to_file(term_count_by_sample, unique_terms, sample_filenames, filename):
    file = open(filename, "w")
    file.write("Samples: ")

    for term in unique_terms[0]:
        file.write("," + term)
    file.write("\n")
    sample_count = 0
    for sample_term_count in term_count_by_sample:
        file.write(sample_filenames[sample_count])
        for term in unique_terms[0]:
            file.write("," + str(sample_term_count[term]))

        file.write("\n")
        sample_count += 1

    file.close()


def read_all_files(filepath):
    data = []
    for filename in os.listdir(filepath):
        data.append(read_file(filepath + "/" + filename))
    return data


def read_file(filename):
    if "metadata" in filename:
        file = open(filename, "r", encoding="iso-8859-1")
    else:
        file = open(filename, "r")
    lines = file.readlines()
    data = []
    for line in lines:
        line = line.replace("\n", "")
        data.append(line.split("\t"))
    file.close()

    del data[0]
    return [data, filename]


def get_terms(term_type, data):
    item_no = get_correct_itemno(term_type)

    term_list = []
    for items in data:
        term_list.append(
            [float(items[0].split("_")[5]), items[item_no].replace('"NA"', '').replace('"', '').split(',')])

    return term_list


def get_unique_terms(term_count_by_sample):
    unique_terms = []
    for sample in term_count_by_sample:
        for term_count in sample:
            unique_terms.append(term_count)
    return set(unique_terms)


def count_unique_terms(data, term_type):
    term_count_dict = get_term_count(data)

    return term_count_dict


def limit_occurrence(term_count_dict, limit):
    minimum_term_count_dict = {}

    for term in term_count_dict:
        if term_count_dict[term] >= limit:
            minimum_term_count_dict[term] = term_count_dict[term]
    return minimum_term_count_dict


def get_term_count(term_list):
    term_count = {}
    for item in term_list:
        coverage = float(item[0])
        for term in item[1]:
            if term in term_count:
                term_count[term] += coverage
            else:
                term_count[term] = coverage

    if '' in term_count:
        del term_count['']
    return term_count


def read_config(file):
    data = open(file, "r")
    settings = {}
    for line in data:
        if line[0] != "#":
            settings[line.split("=")[0]] = line.split("=")[1].replace("\n", "")
    return settings


def get_correct_itemno(term_type):
    if term_type == "GO":
        return 5
    if term_type == "KEGG":
        return 6
    if term_type == "IPR":
        return 3
    print("Invalid term type - selecting IPR as default")
    return 3


if __name__ == '__main__':
    run_preprocessing(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7],
                      sys.argv[8], sys.argv[9])
