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
from multiprocessing import Process, JoinableQueue
import traceback
import sys

# Load librosa for audio processing
from librosa import load, feature

# Load eyed3 to extract mp3 metadata
import eyed3

##################################################
# Set constants and special vars
##################################################
# iTunes directory to loop through
itunes_dir = r'/Users/chiwang/Documents/iTunes 20160601 copy/iTunes Media/Music/'

# Number of workers to parallelize with multiprocessing
num_workers = 20
q = JoinableQueue()
final_results_list = []

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
        y, sr = load(file_path, offset = 45, duration = 30)

        # Extract MFCC and format to be a single array of features (each MFCC is returned as an array, so we get an
        #   array of arrays where we only want a single array)
        mfcc = feature.mfcc(y = y, n_mfcc = 15, fmax = 8000)
        mfcc_flat = flatten_list(mfcc)

        # Return flat list of MFCCs
        return mfcc_flat

##################################################
# Define simple utility functions
##################################################
def flatten_list(list):
    return [item for sublist in list for item in sublist]

##################################################
# Set up queueing and multiprocessing functionality, create worker function processor_job to extract features
##################################################
# Define worker function that generates features of a single song, the multiprocessing engine will feed this worker
#   with songs to analyze in parallel
def processor_job(worker):
    # Retrieve file path from the worker
    file_path = worker[0]

    # Create Song object
    song = Song(file_path)

    # When extracting the MFCC, we first check whether or not the song is longer than 2 minutes (stored in the
    #   Song.flag_to_analyze attribute
    if song.flag_to_analyze:
        song_result = [song.mp3_title, song.mp3_artist, song.mp3_genre, song.mp3_length]
        song_result.extend(song.mfcc_flat)
        final_results_list.append(song_result)

# Define a processor function which manages how many songs to analyze at once by pulling from the queue and processing
#   it via processor_job
def processor():
    while True:
        # Get a worker from the queue
        worker = q.get()
        file = worker[1]
        current_song_counter = worker[2]
        total_songs = worker[3]

        # Run the job with the available worker in the queue
        try:
            print 'Analyzing song #{} / {}: {}'.format(current_song_counter, total_songs, file)
            processor_job(worker)
        except Exception as e:
            traceback.print_exc(file = sys.stdout)
        finally:
            # Job complete
            # print 'Finished analyzing {}'.format(file_path)
            q.task_done()

# Kick off processors of workers
for x in range(num_workers):
    p = Process(target = processor)

    # Classify process as a daemon so it dies when the main thread dies
    p.daemon = True

    # Processor goes into effect
    p.start()

##################################################
# Loop through iTunes library
##################################################
# Loop through iTunes library once to gather total number of songs for our reference
total_songs = 0
for root, dirs, files in os.walk(itunes_dir):
    for file in files:
        # Only look at mp3 files
        if file[-3:] == 'mp3':
            # Add 1 song to total_songs
            total_songs += 1

# Define main function which iterates through iTunes library and adds songs to the multiprocessing queue
current_song_counter = 1
for root, dirs, files in os.walk(itunes_dir):
    for file in files:
        # Format file path
        file_path = os.path.join(root, file)

        # Only look at mp3 files
        if file_path[-3:] == 'mp3':
            # Start multiprocessing and feed file path of song to worker
            q.put([file_path, file, current_song_counter, total_songs])
            current_song_counter += 1

# Wait until all processes terminate before terminating main
q.join()

##################################################
# Save final results
##################################################
# Convert results table (list of lists) into pandas dataframe
pd.DataFrame(final_results_list).to_csv('features.csv')