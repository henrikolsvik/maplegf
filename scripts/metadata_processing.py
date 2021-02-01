import sys


def main(gendermeta_filename, pidmeta_filename, output_filename):
    genders = read_file(gendermeta_filename)
    pid = read_file(pidmeta_filename)

    combined_data = []
    for item in pid:
        for entry in genders:
            if item[0] == entry[0]:
                combined_data.append([item[1], entry[1]])

    write_metadata(combined_data, output_filename)

    return None


def read_file(filename):
    data = []
    file = open(filename, "r")
    lines = file.readlines()
    for line in lines:
        data.append(line.replace("\n", "").split(","))
    file.close()
    return data


def write_metadata(combined_data, output_filename):
    file = open(output_filename, "w")
    for item in combined_data:
        file.write(str(item[0]) + "," + str(item[1]) + "\n")
    file.close()
    return None


main(sys.argv[1], sys.argv[2], sys.argv[3])
