import sys


def run_preprocessing(filename, term_type, minimum_occurrence, output_filename):
    data = read_file(filename)
    terms = count_unique_terms(data, term_type, minimum_occurrence)
    if output_filename is not None:
        write_terms_to_file(terms, output_filename)
    else:
        print(terms)
        print("Items matching criteria: ", len(terms))


def write_terms_to_file(terms, filename):
    file = open(filename, "w")
    for item in terms:
        file.write(item + "\t" + str(terms[item]) + "\n")
    file.close()


def read_file(filename):
    file = open(filename, "r")
    lines = file.readlines()
    data = []
    for line in lines:
        line = line.replace("\n", "")
        data.append(line.split("\t"))
    file.close()

    del data[0]
    return data


def count_unique_terms(data, term_type, limit):
    itemno = get_correct_itemno(term_type)

    termlist = []
    for items in data:
        termlist += items[itemno].replace('"NA"', '').replace('"', '').split(',')

    term_count_dict = get_term_count(termlist)
    term_count_limited = limit_occurrence(term_count_dict, limit)

    return term_count_limited


def limit_occurrence(term_count_dict, limit):

    minimum_term_count_dict = {}

    for term in term_count_dict:
        if term_count_dict[term] >= limit:
            minimum_term_count_dict[term] = term_count_dict[term]
    return minimum_term_count_dict


def get_term_count(termlist):
    term_count = {}
    for item in termlist:
        if item in term_count:
            term_count[item] += 1
        else:
            term_count[item] = 1
    del term_count['']
    return term_count


def get_correct_itemno(term_type):
    if term_type == "GO":
        return 5
    if term_type == "KEGG":
        return 6
    else:
        return 3


if __name__ == '__main__':
    if len(sys.argv) == 5:
        run_preprocessing(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
    run_preprocessing(sys.argv[1], sys.argv[2], int(sys.argv[3]), None)