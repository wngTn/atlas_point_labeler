import struct
import os

import logging
logger = logging.getLogger(__name__)

def read_labels(filename):
    """ read labels from given file. """
    contents = bytes()
    with open(filename, "rb") as f:  # rb = read binary
        f.seek(0, 2)  # move the cursor to the end of the file
        num_points = int(f.tell() / 4)
        f.seek(0, 0)
        contents = f.read()

    arr = [struct.unpack('<I', contents[4 * i:4 * i + 4])[0] for i in range(num_points)]

    return arr

def delete_labels(filename, indices):
    """
    Takes only the labels specified by indices

    :param filename: The filename of the label
    :param indices: The indices to choose
    """
    labels = read_labels(filename)

    labels = [struct.pack('<I', label) for i, label in enumerate(labels) if i in indices]
    os.remove(filename)
    logger.debug(f"Remove {filename}")
    with open(filename, "bw") as f:
        f.write(labels)
    
    logger.debug(f"Written {filename}")
    


def write_labels(filename, labels):
    """ write labels in given file. """
    arr = [struct.pack('<I', label) for label in labels]

    with open(filename, "bw") as f:
        for a in arr:
            f.write(a)


def overwrite_labels(filename, indices, label_num):
    """ Reads the label with filename and overwrites the <indices> with <label_num> """
    labels = read_labels(filename)

    for index in indices:
        labels[index] = label_num

    arr = [struct.pack('<I', label) for label in labels]

    with open(filename, "bw") as f:
        for a in arr:
            f.write(a)

    logger.debug(f"Overwritten {filename} with label: {label_num}. Wrote {len(indices)} label...")
