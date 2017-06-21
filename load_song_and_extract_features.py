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
import os
from multiprocessing import Process, JoinableQueue
import logging
import psutil
import time
import csv

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
# num_workers = 1
q = JoinableQueue()

# Set mp3 analysis constants
min_song_length = 120
librosa_load_offset = 45
librosa_load_duration = 30
num_mfcc_coefficients = 5
max_mfcc_frequency = 8000

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
        return eyed3_mp3_obj.tag.title, \
               eyed3_mp3_obj.tag.artist, \
               eyed3_mp3_obj.tag.genre.name, \
               eyed3_mp3_obj.info.time_secs

    def librosa_mfcc_extract(self, file_path):
        # Load librosa object (only load 30 seconds of the song from 0:45 - 1:15)
        y, sr = load(file_path, offset = librosa_load_offset, duration = librosa_load_duration)

        # Extract MFCC and format to be a single array of features (each MFCC is returned as an array, so we get an
        #   array of arrays where we only want a single array)
        mfcc = feature.mfcc(y = y, n_mfcc = num_mfcc_coefficients, fmax = max_mfcc_frequency)
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
        logger.debug('Analyzed outputs: {}, {}, {}, {}, {}'.format(song.mp3_title, song.mp3_artist, song.mp3_genre, song.mp3_length, len(song.mfcc_flat)))
        song_result = [current_song_counter, num_workers, psutil.cpu_percent(), psutil.virtual_memory()[2], total_time_elapsed, song.mp3_title, song.mp3_artist, song.mp3_genre, song.mp3_length]
        # song_result.extend(song.mfcc_flat[2:])
        with open(output_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(song_result)

# Define a processor function which manages how many songs to analyze at once by pulling from the queue and processing
#   it via processor_job
def processor():
    while True:
        # Get a worker from the queue
        worker = q.get()
        file_path = worker[0]
        file = worker[1]
        current_song_counter = worker[2]
        total_songs = worker[3]

        # Run the job with the available worker in the queue
        try:
            logger.info('Analyzing song #{} / {}: {}'.format(current_song_counter, total_songs, file))
            processor_job(worker)
        except Exception as e:
            logger.exception('Error occurred processing song')
        finally:
            # Job complete
            logger.debug('Finished analyzing {}'.format(file_path))
            q.task_done()

for num_workers in [1, 2, 4, 8, 16, 32, 64]:
    logger.info('Starting iteration with {} number of workers'.format(num_workers))
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
                q.put([file_path, file, current_song_counter, total_songs, num_workers])
                current_song_counter += 1

        if current_song_counter > 200:
            break

    # Wait until all processes terminate before proceeding
    q.join()