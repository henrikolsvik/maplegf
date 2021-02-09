import sys


def main(input_filename, output_filename, type):
    data = read_file(input_filename)
    combined_data = []

    if type == "gender":
        for item in data:
            if item[1] == "M":
                item[1] = 1
            if item[1] == "F":
                item[1] = 0
            combined_data.append([item[0], item[1]])

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
