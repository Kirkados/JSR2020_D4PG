"""
This script saves and loads neural network parameters

@author: Kirk Hovell
"""

import os
import tensorflow as tf

from settings import Settings

class Saver:

    def __init__(self, sess, filename):
        self.sess = sess
        self.filename = filename

    def save(self, n_iteration, policy_input, policy_output):
        # Save all the tensorflow parameters from this session into a file
        # The file is saved to the directory Settings.MODEL_SAVE_DIRECTORY.
        # It uses the n_iteration in the file name

        print("Saving neural networks at iteration number " + str(n_iteration) + "...")

        os.makedirs(os.path.dirname(Settings.MODEL_SAVE_DIRECTORY + self.filename), exist_ok = True)
        self.saver.save(self.sess, Settings.MODEL_SAVE_DIRECTORY + self.filename + "/Iteration_" + str(n_iteration) + ".ckpt")

    def load(self):
        # Try to load in weights to the networks in the current Session.
        # If it fails, or we don't want to load (Settings.RESUME_TRAINING = False)
        # then we start from scratch

        self.saver = tf.train.Saver(max_to_keep = 5) # initialize the tensorflow Saver()

        if Settings.RESUME_TRAINING:
            print("Attempting to load in previously-trained model")
            try:
                ckpt = tf.train.get_checkpoint_state(Settings.MODEL_SAVE_DIRECTORY + Settings.RUN_NAME)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print("Model successfully loaded!")
                return True

            except (ValueError, AttributeError):
                print("No model found... :(")
                return False
        else:
            return False

    def initialize(self):
        self.saver = tf.train.Saver(max_to_keep = 5) # initialize the tensorflow Saver() without trying to load in parameters