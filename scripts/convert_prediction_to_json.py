__author__ = 'Zachary W. Ulissi'
__email__ = 'zulissi@andrew.cmu.edu'

import pickle
import sys
import collections
import json
import bson
import re

# reads in pickled array of objects


def convert(data):  # untested
    if isinstance(data, basestring):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(convert, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    elif isinstance(data, bson.objectid.ObjectId):
        return str(data)
    else:
        return data


def format(data):
    result = []
    header = {"index": {"_index": "structures",
                        "_type": "structure"}}
    order = ['CO', 'H']
    #Handle DFT energies
    for j in range(len(data[0])):
        document = data[0][j][0]
        #Use the calculated energies
        document['energies'] = {}
        document['uncertainties'] = {}
        for i in range(len(data[0][j][1])):
            document['energies'][order[i]] = data[0][j][1][i][0]
            document['uncertainties'][order[i]] = data[0][j][1][i][1]

        document['elements'] = {a[0]: int(a[1]) if a[1] != '' else 1
                                for a in re.findall(r'([A-Z][a-z]*)(\d*)', document['formula'])}

        result.append(header)
        result.append(document)
    #Handle catalog entries
    for j in range(len(data[1])):
        document = data[1][j][0]
        #use the predicted energies
        document['energies'] = {}
        document['uncertainties'] = {}
        for i in range(len(data[1][j][1])):
            document['energies'][order[i]] = data[1][j][1][i][0]
            document['uncertainties'][order[i]] = data[1][j][1][i][1]
        document['elements'] = {a[0]: int(a[1]) if a[1] != '' else 1
                                for a in re.findall(r'([A-Z][a-z]*)(\d*)', document['formula'])}
        result.append(header)
        result.append(document)
    return result


if len(sys.argv) > 1:
    data_file = open(sys.argv[1])                # open the command line argument
    data = pickle.load(data_file)                # load in the data
    data = convert(data)                         # convert it to unicode
    file = open(sys.argv[1][0: -4] + ".json", "w")      # make a new file to output to
    json.dump(format(data), file)
    #file.write(format(data))                     # write in the data as JSON

    data_file.close()
    file.close()
