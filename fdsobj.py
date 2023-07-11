import os
import math
from array import array
import fdsheader
import numpy as np

inf = float('inf')

class FdsObj:
    '''fds file object for reading '''
    __isOpen = False
    __fin = None
    __num_bytes_per_datum = None
    __bins_per_shot_to_read = None
    __bins_per_shot_to_skip = None
    __bytes_per_shot_to_read = None
    __bytes_per_shot_to_skip = None

    path = None
    header = None
    start_bin = None
    end_bin = None
    start_shot = None
    block_size = 1

    def __init__(self, filepath, bin_lims=None, start_shot=0):
        self.path = os.path.normpath(filepath)
        self.header = fdsheader.readheader(filepath)
        self.start_shot = start_shot

        # open file
        if not self.__isOpen:
            try:
                self.__fin = open(self.path,'rb')

                self.__isOpen = True
            except:
                raise ValueError('Could not open file.')

        self.header = fdsheader.readheader(self.path)

        # read needed header values
        try:
            dataStart_byte = self.header['HeaderSize_Bytes']
            self.bin_spacing_m = self.header['DataLocusSpacing_m']
            self.sample_rate_Hz = self.header['TimeStepFrequency_Hz']
            self.first_bin_m = self.header['PositionOfFirstSample_m']
            num_bins = self.header['DataLocusCount']
            num_shots = self.header['TimeStepCount']
            self.duration_s = num_shots / self.sample_rate_Hz
            self.data_type = self.header['DataEncoding']
        except:
            raise ValueError('Missing header values.')

        if bin_lims != None:
            self.start_bin = bin_lims[0]
            self.end_bin = bin_lims[1]
        else:
            self.start_bin = 0
            self.end_bin = num_bins - 1

        if self.start_bin < 0:
            self.start_bin = 0

        if self.end_bin > num_bins-1:
            self.end_bin = num_bins-1

        if self.start_shot < 0:
            self.start_shot = 0

        if self.start_shot > num_shots-1:
            raise ValueError('Start shot past file end.')

        if self.data_type == 'uint16':
            num_bytes_per_datum = 2
        elif self.data_type == 'int16':
            num_bytes_per_datum = 2
        elif self.data_type == 'real32' or self.data_type == 'single':
            num_bytes_per_datum = 4
        else:
            raise ValueError('Unsupported data type: returning.')

        self.__bins_per_shot_to_read = int(self.end_bin - self.start_bin + 1)
        self.__bins_per_shot_to_skip = int(num_bins - self.__bins_per_shot_to_read)

        self.__bytes_per_shot_to_read = self.__bins_per_shot_to_read*num_bytes_per_datum
        self.__bytes_per_shot_to_skip = self.__bins_per_shot_to_skip*num_bytes_per_datum

        aoi_start_byte = int(dataStart_byte +
                             (self.start_shot * num_bins + self.start_bin)*num_bytes_per_datum)

        # it is assumed that this will leave us at the first byte to be read in the desired range in read_shot
        self.__fin.seek(aoi_start_byte,0)

    def __del__(self):
        self.__fin.close()

    # is is assumed that the current pos in the file is the first byte to be read in the current shot in the desired range.
    # __init__ ensures this for the first shot read
    def read_shot(self):
        if self.header['DataEncoding'] == 'uint16':
            read_arr = array('H')
        elif self.header['DataEncoding'] == 'int16':
            read_arr = array('i')
        elif self.header['DataEncoding'] == 'real32' or self.header['DataEncoding'] == 'single':
            read_arr = array('f')
        else:
            raise ValueError('Unsupported data type: returning.')

        try:
            read_arr.fromfile(self.__fin,self.__bins_per_shot_to_read)
            self.__fin.seek(self.__bytes_per_shot_to_skip, 1)
        except:
            raise StopIteration

        return read_arr

    def read(self):
        return np.asarray([shot for i, shot in zip(range(self.block_size), self)])

    def num_bins(self):
        return self.end_bin - self.start_bin + 1

    def __iter__(self):
        return self

    def __next__(self):
        return self.read_shot()

    def __repr__(self):
        return "FDS File: % s\n" \
               "data_type: % s\n" \
               "start_bin: % s\n" \
               "end_bin: % s\n" \
               "start_shot: % s \n" \
               "sample_rate_Hz: % s\n" \
               "bin_spacing_m: % s\n" \
               "block_size: % s" \
               % (os.path.split(self.path)[1],
                  self.data_type,
                  self.start_bin,
                  self.end_bin,
                  self.start_shot,
                  self.sample_rate_Hz,
                  self.bin_spacing_m,
                  self.block_size)
