import sys
import os
import datetime
import numpy as np


def run_preprocessing(sequence_filename, metadata_filepath, sample_output_filename,
                      coverage_key_stats_filename, term_count_unprocessed_filename, term_count_processed_filename,
                      parameter_output_filename, metadata_out_filename, config_file):

    process_values = {"Start_time:": datetime.datetime.now()}

    metadata = read_metadata_file(metadata_filepath)
    config = read_config(config_file)
    term_count_by_sample, coverage_statistics, sequences = [], [], []

    sequence_data = read_file(sequence_filename, config)

    term_list = sequence_data[0][1:]
    del sequence_data[0]
    num_of_sequences = len(sequence_data)

    process_values["Number of samples available: "] = num_of_sequences
    process_values["Number of metadata items: "] = str(len(metadata))

    metadata_out = open(metadata_out_filename, "a+")
    for metadata_item in metadata:
        for sequence in sequence_data:
            if metadata_item[0] in sequence[0] and sequence[0] not in [x[0] for x in sequences]:
                metadata_out.write(str(metadata_item[0]) + "," + str(metadata_item[1]) + "\n")
                sequences.append(sequence)

    metadata_out.close()
    samples = []

    for sequence in sequences:
        samples.append(sequence[0])
        term_count_by_sample.append(count_unique_terms(sequence, term_list))
        # coverage_statistics.append(get_coverage_data(term_list))

    process_values["Number of samples included: "] = str(len(sequence_data))

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
        write_terms_to_file(term_count_by_sample_limited, term_count_by_sample_limited, samples,
                            sample_output_filename)
        write_term_count_overview(term_count_by_sample_limited, term_count_processed_filename)
        write_coverage_key_stats(coverage_statistics, sequence_filename, coverage_key_stats_filename)

        process_values["End_time:"] = datetime.datetime.now()
        elapsed_time = process_values["End_time:"] - process_values["Start_time:"]
        process_values["Total_time"] = elapsed_time.total_seconds()
        process_values["End_time:"] = str(process_values["End_time:"])
        process_values["Start_time:"] = str(process_values["Start_time:"])
        print(process_values)

        write_preprocessing_parameter_data(parameter_output_filename, sample_output_filename, num_of_sequences,
                                           process_values, config)


def write_preprocessing_parameter_data(parameter_output_filename, sample_output_filename, num_of_sequences,
                                       process_values, config):
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


def get_sequence_file_list(sequence_dir, metadata):
    file_list = []
    for sequence in metadata:
        for filename in os.listdir(sequence_dir):
            if sequence[0] in filename:
                file_list.append(filename)
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


def read_file(filename, config):
    file = open(filename, "r")
    lines = file.readlines()
    file.close()
    data = []
    for i in range(0, len(lines)):
        if not bool(config["include_mapped_and_unmapped"]):
            data.append(lines[i].replace("\n", "").split("\t"))
        else:
            if i != 1 and i != 2:
                data.append(lines[i].replace("\n", "").split("\t"))

    return np.transpose(np.array(data)).tolist()


def get_unique_terms(term_count_by_sample):
    unique_terms = []
    for sample in term_count_by_sample:
        for term_count in sample:
            unique_terms.append(term_count)
    return set(unique_terms)


def count_unique_terms(data, sorted_terms):
    term_count_dict = {}
    for i in range(0, len(sorted_terms)):
        term_count_dict[sorted_terms[i]] = float(data[i + 1])

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


if __name__ == '__main__':
    run_preprocessing(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7],
                      sys.argv[8], sys.argv[9])
