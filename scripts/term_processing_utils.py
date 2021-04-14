

def find_sequences_matching_metadata_and_write_matching_metadata(sequence_data, metadata, metadata_out_filename):
    sequences = []
    metadata_out = open(metadata_out_filename, "a+")
    for metadata_item in metadata:
        for sequence in sequence_data:
            if metadata_item[0] in sequence[0] and sequence[0] not in [x[0] for x in sequences]:
                metadata_out.write(str(metadata_item[0]) + "," + str(metadata_item[1]) + "\n")
                sequences.append(sequence)
    metadata_out.close()
    return sequences

