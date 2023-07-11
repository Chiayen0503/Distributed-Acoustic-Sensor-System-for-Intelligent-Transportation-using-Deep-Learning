import os
import math

def isint(x):
    try:
        y = int(x)
        return True
    except ValueError:
        return False

def isfloat(x):
    try:
        y = float(x)
        return True
    except ValueError:
        return False

def isvector(x):
    return '[' in x  and ']' in x


def parsevalue(value):
    if isint(value):
        return int(value)

    if isfloat(value) and not isint(value):
        return float(value)

    if isvector(value):
        value = value.replace('[', '')
        value = value.replace(']', '')
        value = value.replace(',', '')
        value = value.strip()
        value = value.split(" ")

        vec = []

        for x in value:
            y = parsevalue(x)
            vec.append(y)

        return vec

    # default to string, but strip quotes and whitespace
    y = value
    y = y.replace('\"', '')
    y = y.replace('\'', '')
    y = y.strip()
    return y

def getline(file):
    line = next(file)
    return line.decode(errors='ignore')

def readheader(file_path, parse_header=True):

    header = dict()
    bytes_read = 0

    file_path = os.path.normpath(file_path)
    file = open(file_path,'rb')

    #line 1 is FDS_Version = 1
    line = getline(file)
    bytes_read += len(line)
    key, value = [x.strip() for x in line.split('=')]
    FDS_Version = parsevalue(value)
    if parse_header:
        value = FDS_Version
    header[key] = value

    #line 2 is HeaderSize_Bytes = size of header in bytes
    line = getline(file)
    bytes_read += len(line)
    key, value = [x.strip() for x in line.split('=')]
    HeaderSize_Bytes = parsevalue(value)
    if parse_header:
        value = HeaderSize_Bytes
    header[key] = value

    #line 3 is HeaderSectionSizes_Bytes = size of header sections in bytes
    line = getline(file)
    bytes_read += len(line)
    key, value = [x.strip() for x in line.split('=')]
    HeaderSectionSizes_Bytes = parsevalue(value)
    if parse_header:
        value = HeaderSectionSizes_Bytes
    header[key] = value

    #line 4 is HeaderSectionLabels = names of header sections
    line = getline(file)
    bytes_read += len(line)
    key, value = [x.strip() for x in line.split('=')]
    HeaderSectionLabels = parsevalue(value)
    if parse_header:
        value = HeaderSectionLabels
    header[key] = value

    if FDS_Version != 1:
        raise ValueError('Only version 1 fds files currently supported!')

    #count the number of bytes in the header for reading
    num_bytes_no_padding = sum([HeaderSectionSizes_Bytes[i]
        for i in range(len(HeaderSectionSizes_Bytes))
        if not HeaderSectionLabels[i] == 'Padding'])

    #read the rest of the header
    while True:
        line = getline(file)
        bytes_read += len(line)
        if bytes_read >= num_bytes_no_padding:
            break
        key, value = [x.strip() for x in line.split('=')]
        if parse_header:
            value = parsevalue(value)
        header[key] = value

    return header

    file.close()
