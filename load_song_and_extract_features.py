# This script loops through my iTunes library and performs the following for each song and saves overall results in
#   a file on S3
#       - Loads each mp3
#       - Extracts features
#           - Extracts the first 15 sets of MFCC coefficients
#           - Extracts the tempo
#           - Extracts additional metadata (length, genre, song name, artist)
#       - Stores above information in arrays

##################################################
# Load libraries
##################################################
# Load general libraries
import numpy as np
import pandas as pd
import os
import multiprocessing

# Load librosa for audio processing
import librosa

# Load eyed3 to extract mp3 metadata
import eyed3

##################################################
# Set constants
##################################################
# iTunes directory to loop through
itunes_dir = r'/Users/chiwang/Documents/iTunes 20160601 copy/iTunes Media/Music/'

##################################################
# class Song represents each mp3 being analyzed
##################################################
class Song:
    def __init__(self, file_path):
        # Extract mp3 metadata with eyed3 library
        self.mp3_title, \
        self.mp3_artist, \
        self.mp3_genre, \
        self.mp3_length = self.eyed3_mp3_metadata_extract(file_path)

        # We are only interested in a song if it's longer than 2 minutes (otherwise it's likely an intro / interlude,
        #   which we don't want to analyze)
        if self.mp3_length >= 120:
            # Set flag so we can easily tell if this file is to be analyzed in the main script
            self.flag_to_analyze = True

            # Extract MFCC vectors
            self.mfcc_flat = self.librosa_mfcc_extract(file_path)
        else:
            # Set flag so we can easily tell if this file is to be analyzed in the main script
            self.flag_to_analyze = False

    def eyed3_mp3_metadata_extract(self, file_path):
        # Load mp3 eyed3 object
        eyed3_mp3_obj = eyed3.load(unicode(file_path, 'utf-8'))

        # Return metadata
        return eyed3_mp3_obj.tag.title, \
               eyed3_mp3_obj.tag.artist, \
               eyed3_mp3_obj.tag.genre.name, \
               eyed3_mp3_obj.info.time_secs

    def librosa_mfcc_extract(self, file_path):
        # Load librosa object (only load 30 seconds of the song from 0:45 - 1:15)
        y, sr = librosa.load(file_path, offset = 45, duration = 30)

        # Extract MFCC and format to be a single array of features (each MFCC is returned as an array, so we get an
        #   array of arrays where we only want a single array)
        mfcc = librosa.feature.mfcc(y = y, n_mfcc = 15, fmax = 8000)
        mfcc_flat = [item for sublist in mfcc for item in sublist]

        # Return flat list of MFCCs
        return mfcc_flat


##################################################
# Loop through iTunes library
##################################################
for root, dirs, files in os.walk(itunes_dir):
    for file in files:
        # Format file path
        file_path = os.path.join(root, file)

        # Only look at mp3 files
        if file_path[-3:] == 'mp3':
            # Create Song object
            song = Song(file_path)
            print song.mp3_artist, song.mp3_title, song.mp3_genre, song.mp3_length, len(song.mfcc_flat)