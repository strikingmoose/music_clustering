# This script loops through my iTunes library and performs the following for each song and saves overall results in
#   a csv file in the project root directory
#       - Loads each mp3
#       - Extracts features
#           - Extracts a set amount of MFCCs
#           - Extracts the tempo
#           - Extracts additional metadata (length, genre, song name, artist)
#       - Stores above information in arrays

##################################################
# Load libraries
##################################################
# Load general libraries
import os
from multiprocessing import Process, JoinableQueue, Value, Lock
import logging
import psutil
import time
import csv
from sklearn import preprocessing

# Load librosa for audio processing
from librosa import load, feature

# Load eyed3 to extract mp3 metadata
import eyed3

##################################################
# Set constants and special vars
##################################################
# iTunes directory to loop through
itunes_dir = r'/Users/chiwang/Documents/iTunes 20160601 copy/iTunes Media/Music/'
output_csv = r'features.csv'

# Number of workers to parallelize with multiprocessing
num_workers = 2
q = JoinableQueue()

# Set mp3 analysis constants
min_song_length = 120
librosa_load_offset = 45
librosa_load_duration = 15
num_mfcc_coefficients = 10
max_mfcc_frequency = 8000

# The error counter tracks the number of errors that occured within a process that caused the process to die (e.g.
#   unicode encoding error), we have to use the Value() and Lock() functions from multiprocessing to ensure a shared
#   counter is being successfully updated by all processes simultaneously
error_counter = Value('i', 0)
lock = Lock()

##################################################
# Set up logging config
##################################################
# Set up main logger object
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Prevent main logger from logging to console, only use stream handlers below
logger.propagate = False

# Set message formatting
formatter = logging.Formatter('%(asctime)s || %(name)s || %(levelname)s || %(message)s')

# Set stream_handler for console logging and add to main logger
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Configure eyed3 log (very verbose)
eyed3.log.setLevel("ERROR")

##################################################
# Clear output csv before writing
##################################################
open(output_csv, 'w').close()

##################################################
# Class Song represents each mp3 being analyzed
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
        if self.mp3_length >= min_song_length:
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
        return eyed3_mp3_obj.tag.title.encode('utf-8'), \
               eyed3_mp3_obj.tag.artist.encode('utf-8'), \
               eyed3_mp3_obj.tag.genre.name, \
               eyed3_mp3_obj.info.time_secs

    def librosa_mfcc_extract(self, file_path):
        # Load librosa object (only load 30 seconds of the song from 0:45 - 1:15)
        y, sr = load(file_path, offset = librosa_load_offset, duration = librosa_load_duration)

        # Extract MFCC and format to be a single array of features (each MFCC is returned as an array, so we get an
        #   array of arrays where we only want a single array)
        mfcc = feature.mfcc(y = y, n_mfcc = num_mfcc_coefficients, fmax = max_mfcc_frequency).T

        # Scale the coefficients to mean zero and unit variance
        mfcc_scaled = preprocessing.StandardScaler().fit_transform(mfcc)
        logger.debug('Transposed MFCC shape is {}'.format(mfcc.shape))

        # Return flat list of MFCCs
        return mfcc_scaled

##################################################
# Set up queueing and multiprocessing functionality, create worker function processor_job to extract features
##################################################
# Define worker function that generates features of a single song, the multiprocessing engine will feed this worker
#   with songs to analyze in parallel
def processor_job(worker):
    # Start timer to measure time it takes to run job
    start_time = time.time()

    # Retrieve file path from the worker
    file_path = worker[0]
    current_song_counter = worker[2]
    num_workers = worker[4]

    # Create Song object
    song = Song(file_path)

    # When extracting the MFCC, we first check whether or not the song is longer than 2 minutes (stored in the
    #   Song.flag_to_analyze attribute
    if song.flag_to_analyze:
        # Capture timer value
        total_time_elapsed = time.time() - start_time

        # As songs are analyzed, save the results in a text file
        # logger.debug('Analyzed outputs: {}, {}, {}, {}, {}'.format(song.mp3_title, song.mp3_artist, song.mp3_genre, song.mp3_length, len(song.mfcc_flat)))
        for mfcc_array in song.mfcc_flat:
            song_result = [current_song_counter, song.mp3_title, song.mp3_artist, song.mp3_genre]
            song_result.extend([ '%.5f' % x for x in mfcc_array])
            with open(output_csv, 'a') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(song_result)
        # song_result = [current_song_counter, num_workers, psutil.cpu_percent(), psutil.virtual_memory()[2], total_time_elapsed, song.mp3_title, song.mp3_artist, song.mp3_genre, song.mp3_length]

# Define a processor function which manages how many songs to analyze at once by pulling from the queue and processing
#   it via processor_job
def processor(error_counter, lock):
    while True:
        # Get a worker from the queue
        worker = q.get()
        file_path = worker[0]
        file = worker[1]
        current_song_counter = worker[2]
        total_songs = worker[3]

        # Run the job with the available worker in the queue
        try:
            logger.info('Analyzing song #{} / {}: {} ({}% CPU, {}% MEM, {} errors so far)'.format(current_song_counter, total_songs, file, psutil.cpu_percent(), psutil.virtual_memory()[2], error_counter.value))
            processor_job(worker)
        except Exception as e:
            # If an exception happened inside a process, we track the errors and increment the counter before exiting
            with lock:
                error_counter.value += 1
            logger.exception('Error occurred processing song {}'.format(file))
        finally:
            # Job complete
            logger.debug('Finished analyzing {}'.format(file_path))
            q.task_done()

# Kick off processors of workers
for x in range(num_workers):
    p = Process(target = processor, args = (error_counter, lock))

    # Classify process as a daemon so it dies when the main thread dies
    p.daemon = True

    # Processor goes into effect
    p.start()

##################################################
# Finally, the main function looping through iTunes library
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
            q.put([file_path, file, current_song_counter, total_songs, num_workers])
            current_song_counter += 1

    # if current_song_counter > 200:
    #     break

# Wait until all processes terminate before proceeding
q.join()