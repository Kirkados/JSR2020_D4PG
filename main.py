"""
Main script that runs the D4PG learning algorithm
(https://arxiv.org/pdf/1804.08617)

It features the standard DDPG algorithm with a number
of improvements from other researchers.
Namely:
    Distributed rollouts           (https://arxiv.org/pdf/1602.01783)
    A distributional learner       (http://arxiv.org/abs/1707.06887)
    N-step returns                 (https://arxiv.org/pdf/1602.01783)
    Prioritized experience replay  (http://arxiv.org/abs/1511.05952)

This implementation does not use the
ApeX framework (https://arxiv.org/abs/1803.00933) as the original authors did.
Instead, it uses python's 'threading' and 'multiprocessing' library.

Different tasks are contained in different threads. Tensorflow is thread-safe and automatically multi-threaded.
Each instance of the environment is contained in a different process due to scipy not being thread-safe.

===== Notes =====
No notes at the moment

@author: Kirk Hovell (khovell@gmail.com)

Special thanks to:
    - msinto93 (https://github.com/msinto93)
    - SuReLI   (https://github.com/SuReLI)
    - DeepMind (https://github.com/deepmind)
    - OpenAI   (https://github.com/openai)
    for publishing their codes! The open-source attitude of AI research is wonderful.

Code started: October 15, 2018
Learning algorithm complete: May 6, 2019
"""

# Importing libraries & other classes
# Others'
import shutil
import os
import glob
import time
import threading
import multiprocessing
import random
import datetime
import psutil
import tensorflow as tf
import numpy as np

# My own
from agent import Agent
from learner import Learner
from replay_buffer import ReplayBuffer
from prioritized_replay_buffer import PrioritizedReplayBuffer
from settings import Settings
import saver

#%%
##########################
##### SETTING UP RUN #####
##########################
start_time = time.time()

# Clearing Tensorflow graph
tf.reset_default_graph()

# Setting Tensorflow configuration parameters
config = tf.ConfigProto()
config.intra_op_parallelism_threads = psutil.cpu_count(logical = False) # Number of CPU physical cores recommended
if psutil.cpu_count(logical = False) == 32:
    config.inter_op_parallelism_threads = 32 # RCDC has 32 sockets
else:
    config.inter_op_parallelism_threads = 1 # All my other computers have 1

# Set random seeds
tf.set_random_seed(Settings.RANDOM_SEED)
np.random.seed(Settings.RANDOM_SEED)
random.seed(Settings.RANDOM_SEED)


############################################################
##### New run or continuing a partially completed one? #####
############################################################
# If we're continuing a run
if Settings.RESUME_TRAINING:
    filename                  = Settings.RUN_NAME # Reuse the name too
    starting_episode_number   = np.zeros(Settings.NUMBER_OF_ACTORS, dtype = np.int8) # initializing
    starting_iteration_number = 0 # initializing

    try:
        # Grab the tensorboard path
        old_tensorboard_filename = [i for i in os.listdir(Settings.MODEL_SAVE_DIRECTORY + filename) if i.endswith(Settings.TENSORBOARD_FILE_EXTENSION)][0]

        # For every entry in the tensorboard file
        for tensorboard_entry in tf.train.summary_iterator(Settings.MODEL_SAVE_DIRECTORY + filename + "/" + old_tensorboard_filename):
            # Search each one for the Loss value so you can find the final iteration number
            for tensorboard_value in tensorboard_entry.summary.value:
                if tensorboard_value.tag == 'Logging_Learning/Loss':
                    starting_iteration_number = max(tensorboard_entry.step, starting_iteration_number)

            # Search also for the actors so you can find what episode they were on
            for agent_number in range(Settings.NUMBER_OF_ACTORS):
                for tensorboard_value in tensorboard_entry.summary.value:
                    if tensorboard_value.tag == 'Agent_' + str(agent_number + 1) + '/Number_of_timesteps':
                        starting_episode_number[agent_number] = max(tensorboard_entry.step, starting_episode_number[agent_number])

    except:
        # If the load failed... quit run
        print("Couldn't load in old tensorboard file! Quitting run.")
        raise SystemExit

else: # Otherwise, we are starting from scratch
    # Generate a filename using Settings.RUN_NAME with the current timestamp
    filename                  = Settings.RUN_NAME + '-{:%Y-%m-%d_%H-%M}'.format(datetime.datetime.now())
    starting_episode_number   = np.ones(Settings.NUMBER_OF_ACTORS, dtype = int) # All actors start at episode 0
    starting_iteration_number = 1 # learner starts at iteration 0

# Generate writer that will log Tensorboard scalars & graph
writer = tf.summary.FileWriter(Settings.MODEL_SAVE_DIRECTORY + filename, filename_suffix = Settings.TENSORBOARD_FILE_EXTENSION)

# Saving a copy of the all python files used in this run, for reference
# Make directory if it doesn't already exist
os.makedirs(os.path.dirname(Settings.MODEL_SAVE_DIRECTORY + filename + '/code/'), exist_ok=True)
for each_file in glob.glob('*.py'):
    shutil.copy2(each_file, Settings.MODEL_SAVE_DIRECTORY + filename + '/code/')

#######################################
##### Starting Tensorflow session #####
#######################################
with tf.Session(config = config) as sess:
    print("\nThis run is named " + filename)
    print("\nThe environment file is: environment_" + Settings.ENVIRONMENT + '\n')
    if Settings.TEST_ON_DYNAMICS:
        print("At test time, full dynamics are being used\n")
    else:
        print("At test time, kinematics are being used\n")

    if Settings.KINEMATIC_NOISE:
        print("Noise is being applied to the kinematics during training to simulate a poor controller\n")

    ##############################
    ##### Initializing items #####
    ##############################

    # Initializing saver class (for loading & saving data)
    saver = saver.Saver(sess, filename)

    # Initializing replay buffer, with the option of a prioritized replay buffer
    if Settings.PRIORITY_REPLAY_BUFFER:
        replay_buffer = PrioritizedReplayBuffer()
    else:
        replay_buffer = ReplayBuffer()

    # Initializing thread & process list
    threads = []
    environment_processes = []

    # Event()s are used to communicate with threads while they run.
    # In this case, it is used to signal to the threads when it is time to stop gracefully.
    stop_run_flag           = threading.Event() # Flag to stop all threads
    replay_buffer_dump_flag = threading.Event() # Flag to pause data writing to the replay buffer
    replay_buffer_dump_flag.set() # Set the flag to initially be True so that the agents will write data

    # Generating the learner and assigning it to a thread
    if Settings.USE_GPU_WHEN_AVAILABLE:
        # Allow GPU use when appropriate
        learner = Learner(sess, saver, replay_buffer, writer)
        # Generate the queue responsible for communicating with the agent (for test distribution calculating)
        agent_to_learner, learner_to_agent = learner.generate_queue()
    else:
        # Forcing to the CPU only
        with tf.device('/device:CPU:0'):
            learner = Learner(sess, saver, replay_buffer, writer)
            # Generate the queue responsible for communicating with the agent (for test distribution calculating)
            agent_to_learner, learner_to_agent = learner.generate_queue()
    threads.append(threading.Thread(target = learner.run, args = (stop_run_flag, replay_buffer_dump_flag, starting_iteration_number)))

    # Generating the actors and placing them into their own threads
    for i in range(Settings.NUMBER_OF_ACTORS):
        if Settings.USE_GPU_WHEN_AVAILABLE:
            # Allow GPU use when appropriate
            # Make an instance of the environment which will be placed in its own process
            environment_file = __import__('environment_' + Settings.ENVIRONMENT)
            if Settings.ENVIRONMENT == 'gym':
                environment = environment_file.Environment(filename, i+1, Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES, Settings.VIDEO_RECORD_FREQUENCY, Settings.MODEL_SAVE_DIRECTORY) # Additional parameters needed for gym
            else:
                environment = environment_file.Environment()
            # Set the environment seed
            environment.seed(Settings.RANDOM_SEED*(i+1))
            # Generate the queue responsible for communicating with the agent
            agent_to_env, env_to_agent = environment.generate_queue()
            # Generate the actor
            actor = Agent(sess, i+1, agent_to_env, env_to_agent, replay_buffer, writer, filename, learner.actor.parameters, agent_to_learner, learner_to_agent)

        else:
            with tf.device('/device:CPU:0'):
                # Forcing to the CPU only
                # Make an instance of the environment which will be placed in its own process
                environment_file = __import__('environment_' + Settings.ENVIRONMENT)
                if Settings.ENVIRONMENT == 'gym':
                    environment = environment_file.Environment(filename, i+1, Settings.CHECK_GREEDY_PERFORMANCE_EVERY_NUM_EPISODES, Settings.VIDEO_RECORD_FREQUENCY, Settings.MODEL_SAVE_DIRECTORY) # Additional parameters needed for gym
                else:
                    environment = environment_file.Environment()
                # Set the environment seed
                environment.seed(Settings.RANDOM_SEED*(i+1))
                # Generate the queue responsible for communicating with the agent
                agent_to_env, env_to_agent = environment.generate_queue()
                # Generate the actor
                actor = Agent(sess, i+1, agent_to_env, env_to_agent, replay_buffer, writer, filename, learner.actor.parameters, agent_to_learner, learner_to_agent)

        # Add thread and process to the list
        threads.append(threading.Thread(target = actor.run, args = (stop_run_flag, replay_buffer_dump_flag, starting_episode_number)))
        environment_processes.append(multiprocessing.Process(target = environment.run, daemon = True)) # daemon ensures process is killed when main ends

    # If desired, try to load in partially-trained parameters
    if Settings.RESUME_TRAINING == True:
        if not saver.load():
            # If loading was not successful -> quit program
            print("Could not load in parameters... quitting program")
            raise SystemExit
    else:
        # Don't try to load in parameters, just initialize them instead
        # Initialize saver
        saver.initialize()
        # Initialize Tensorflow variables
        sess.run(tf.global_variables_initializer())


    # Starting all environments
    for each_process in environment_processes:
        each_process.start()

    #############################################
    ##### STARTING EXECUTION OF ALL THREADS #####
    #############################################
    #                                           #
    #                                           #
    for each_thread in threads:                 #
    #                                           #
        each_thread.start()                     #
    #                                           #
    #                                           #
    #############################################
    ############## THREADS STARTED ##############
    #############################################

    # Write the Tensorflow computation graph to file, now that it has been fully built
    writer.add_graph(sess.graph)
    print('Done starting!')


    ####################################################
    ##### Waiting until all threads have completed #####
    ####################################################
    print("Running until threads finish or until you press Ctrl + C")
    try:
        while True:
            time.sleep(0.5)
            # If all threads have ended on their own
            if not any(each_thread.is_alive() for each_thread in threads):
                print("All threads ended naturally.")
                break
    except KeyboardInterrupt: # if someone pressed Ctrl + C
        print("Interrupted by user!")
        print("Stopping all the threads!!")
        # Gracefully stop all threads, ending episodes and saving data
        stop_run_flag.set()
        # Join threads (suspends main.py until threads finish)
        for each_thread in threads:
            each_thread.join()

    print("This run completed in %.3f hours." %((time.time() - start_time)/3600))
    print("Done closing! Goodbye :)")
