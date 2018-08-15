from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import math
import sys
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from optparse import OptionParser
from models.basic_files.dataset_iterator import *
from models.inference_model import *
from read_config import *
import os

class run_model:

    def __init__(self, wd, bA, config = None):

        """ The model is initializer with the hyperparameters.

            Args:
                config : Config() obeject for the hyperparameters.
        """

        # Use default hyperparameters
        if config is None:
            config = Config()

        self.config  = config
        self.model   = bA

        # Vocabulary and datasets are initialized.
        self.dataset = PadDataset(wd, self.config.embedding_size, diff_vocab = self.config.config_dir["diff_vocab"], embedding_path = self.config.config_dir["embedding_path"],\
				  limit_encode = self.config.config_dir["limit_encode"], limit_decode = self.config.config_dir["limit_decode"])


    def add_placeholders(self):

        """ Generate placeholder variables to represent input tensors
        """

        self.encode_input_placeholder  = tf.placeholder(tf.int32, shape=(self.config.config_dir["max_sequence_length_content"], None), name ='encode')
        self.decode_input_placeholder  = tf.placeholder(tf.int32, shape=(self.config.config_dir["max_sequence_length_title"], None),   name = 'decode')
        self.query_input_placeholder   = tf.placeholder(tf.int32, shape=(self.config.config_dir["max_sequence_length_query"], None),   name = 'query')
        self.label_placeholder         = tf.placeholder(tf.int32, shape=(self.config.config_dir["max_sequence_length_title"], None),   name = 'labels')
        self.weights_placeholder       = tf.placeholder(tf.int32, shape=(self.config.config_dir["max_sequence_length_title"], None),   name = 'weights')
        self.feed_previous_placeholder = tf.placeholder(tf.bool, name='feed_previous')
        self.encode_sequence_length    = tf.placeholder(tf.int64, shape=None, name="encode_seq_length")
        self.query_sequence_length     = tf.placeholder(tf.int64, shape=None, name="query_seq_length")
        #Could be used for dynamic padding
        #self.max_content_per_batch_p   = tf.placeholder(tf.int32, name='max_content')
        #self.max_title_per_batch_p     = tf.placeholder(tf.int32, name='max_title')


    def fill_feed_dict(self, encoder_inputs, decoder_inputs, labels, query, weights, encode_seq_length, query_seq_length, 
                      feed_previous=False):

        """ Fills the feed_dict for training at a given time_step.

            Args:
                encode_inputs : Encoder  sequences
                decoder_inputs : Decoder sequences
                labels : Labels for the decoder
                feed_previous : Whether to pass previous state output to decoder.

            Returns:
                feed_dict : the dictionary created.
        """

        feed_dict = {
        self.encode_input_placeholder : encoder_inputs,
        self.decode_input_placeholder : decoder_inputs,
        self.label_placeholder        : labels,
        self.query_input_placeholder  : query, 
        self.weights_placeholder      : weights,
        self.feed_previous_placeholder: feed_previous,
        self.query_sequence_length    : query_seq_length, 
        self.encode_sequence_length   : encode_seq_length,
        }

        return feed_dict


    def run_epoch(self, epoch_number, sess, fp = None):

        """ Defines the per epoch run of the model

            Args:
                epoch_number: The current epoch number
                sess:       :  The current tensorflow session.

            Returns
                total_loss : Value of loss per epoch

        """

        start_time = time.time()
        steps_per_epoch = int(math.ceil(float(self.dataset.datasets["train"].number_of_samples) / float(self.config.batch_size)))

        total_loss = 0

        for step in xrange(steps_per_epoch):

            train_content, train_title, train_labels, train_query, train_weights, train_content_seq, train_query_seq, \
            max_content, max_title,max_query = self.dataset.next_batch(
                self.dataset.datasets["train"],self.config.batch_size, True)



            """ Pass the decoder_inputs for the earlier epochs. As the model
                is trained, the outputs from the previous state should be fed
                to better train the model.
            """
            if (fp is None):
                if(epoch_number > 5):
                    feed_previous = True
                else:
                    feed_previous = False

            else:
                feed_previous = fp

            # Feed the placeholders with encoder_inputs,decoder_inputs,decoder_labels
            feed_dict = self.fill_feed_dict(train_content, train_title, train_labels, train_query, train_weights,
                                            train_content_seq, train_query_seq, feed_previous)

            #Minimize the loss
            _, loss_value, outputs  = sess.run([self.train_op, self.loss_op, self.logits], feed_dict=feed_dict)
            total_loss  += loss_value

            duration = time.time() - start_time

            # Check the loss with forward propogation
            if (step + 1 == steps_per_epoch ) or ((step  + 1) % self.config.config_dir["print_frequency"] == 0):

                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                sys.stdout.flush()

                # Evaluate against the training set.
                print('Training Data Eval:')
                self.print_titles(sess, self.dataset.datasets["train"], 2)
                    
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                self.print_titles(sess,self.dataset.datasets["valid"], 2)
                    
                sys.stdout.flush()

        return float(total_loss)/ float(steps_per_epoch)


    def do_eval(self,sess, data_set):

        """ Does a forward propogation on the data to know how the model's performance is.
             This will be mainly used for valid and test dataset.

            Args:
                sess : The current tensorflow session
                data_set : The datset on which this should be evaluated.

            Returns
                Loss value : loss value for the given dataset.
        """  

        total_loss = 0
        steps_per_epoch =  int(math.ceil(float(data_set.number_of_samples) / float(self.config.batch_size)))

        for step in xrange(steps_per_epoch): 
            train_content, train_title, train_labels, train_query, train_weights, train_content_seq, train_query_seq, max_content, max_title, max_query = self.dataset.next_batch(
                data_set,self.config.batch_size, False)
            
            feed_dict  = self.fill_feed_dict(train_content, train_title, train_labels, train_query, train_weights,
                                             train_content_seq, train_query_seq, feed_previous = True)
            loss_value = sess.run(self.loss_op, feed_dict=feed_dict)
            total_loss += loss_value

        return float(total_loss)/float(steps_per_epoch)



    def print_titles_in_files(self, sess, data_set):

        """ Prints the titles for the requested examples.

            Args:
                sess: Running session of tensorflow
                data_set : Dataset from which samples will be retrieved.
                total_examples: Number of samples for which title is printed.

        """
        total_loss = 0
        f1 = open(self.config.outdir + data_set.name + "_final_results", "wb")

        steps_per_epoch =  int(math.ceil(float(data_set.number_of_samples) / float(self.config.batch_size)))

        for step in xrange(steps_per_epoch):
            train_content, train_title, train_labels, train_query, train_weights, train_content_seq, train_query_seq,\
            max_content, max_title, max_query = self.dataset.next_batch(data_set,self.config.batch_size, False)

            feed_dict = self.fill_feed_dict(train_content, train_title, train_labels, train_query, train_weights,
                                            train_content_seq, train_query_seq, feed_previous = True)

            _decoder_states_ = sess.run(self.logits, feed_dict=feed_dict)

            # Pack the list of size max_sequence_length to a tensor
            decoder_states = np.array([np.argmax(i,1) for i in _decoder_states_])

            # tensor will be converted to [batch_size * sequence_length * symbols]
            ds = np.transpose(decoder_states)
            true_labels = np.transpose(train_labels)
            # Converts this to a length of batch sizes
            final_ds = ds.tolist()
            true_labels = true_labels.tolist()

            for i, states in enumerate(final_ds):

                # Get the index of the highest scoring symbol for each time step
                s =  self.dataset.decode_to_sentence(states)
                t =  self.dataset.decode_to_sentence(true_labels[i])
                f1.write(s + "\n")
                f1.write(t +"\n")

    def print_titles(self, sess, data_set, total_examples):

        """ Prints the titles for the requested examples.

            Args:
                sess: Running session of tensorflow
                data_set : Dataset from which samples will be retrieved.
                total_examples: Number of samples for which title is printed.

        """
        train_content, train_title, train_labels, train_query, train_weights, train_content_seq, train_query_seq,\
        max_content, max_title, max_query = self.dataset.next_batch(data_set, total_examples, False)

        feed_dict = self.fill_feed_dict(train_content, train_title, train_labels, train_query, train_weights, train_content_seq, 
                                        train_query_seq, feed_previous = True)

        _decoder_states_ = sess.run(self.logits, feed_dict=feed_dict)

        # Pack the list of size max_sequence_length to a tensor
        decoder_states = np.array([np.argmax(i,1) for i in _decoder_states_])

        ds = np.transpose(decoder_states)
        true_labels = np.transpose(train_labels)

        # Converts this to a length of batch size
        final_ds = ds.tolist()
        true_labels = true_labels.tolist()

        for i,states in enumerate(final_ds):

            # Get the index of the highest scoring symbol for each time step
            print ("Title is " + self.dataset.decode_to_sentence(states))
            print ("True Summary is " + self.dataset.decode_to_sentence(true_labels[i]))


    def run_training(self):

        """ Train the graph for a number of epochs 
        """

        with tf.Graph().as_default():

            tf.set_random_seed(1357)
            len_vocab = self.dataset.length_vocab_encode()
            initial_embeddings = self.dataset.vocab.embeddings

            self.add_placeholders()

            # Build a Graph that computes predictions from the inference model.
            self.logits = self.model.inference(self.config,
                                               self.config.config_dir["cell_encoder"],
                                               self.config.config_dir["cell_decoder"],
                                               self.encode_input_placeholder,
                                               self.decode_input_placeholder, 
                                               self.query_input_placeholder,
                                               self.config.config_dir["embedding_size"],
                                               self.feed_previous_placeholder,
                                               len_vocab,
                                               self.config.config_dir["hidden_size"],
                                               weights = self.weights_placeholder,
                                               encoder_sequence_length = self.encode_sequence_length,
                                               query_sequence_length = self.query_sequence_length,
                                               initial_embedding = initial_embeddings,
                                               embedding_trainable=self.config.config_dir["embedding_trainable"])

            # Add to the Graph the Ops for loss calculation.
            self.loss_op = self.model.loss_op(self.logits, self.label_placeholder, self.weights_placeholder)

            # Add to the Graph the Ops that calculate and apply gradients.
            self.train_op = self.model.training(self.loss_op, self.config.config_dir["learning_rate"])

         
            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver()

            # Create a session for running Ops on the Graph.
            sess = tf.Session()

            outdir = self.config.config_dir["outdir"]
            # Instantiate a SummaryWriter to output summaries and the Graph.
            summary_writer = tf.train.SummaryWriter(outdir + "Logs" ,sess.graph)

            saver.restore(sess, self.config.outdir + 'best_model')
            test_loss = self.do_eval(sess, self.dataset.datasets["test"])

            print ("Test Loss:{}".format(test_loss))
            self.print_titles_in_files(sess, self.dataset.datasets["test"])

def main():
    c = Config(sys.argv[1])
    run_attention = run_model(c.config_dir["working_dir"], BasicAttention(), c)
    run_attention.run_training()

if __name__ == '__main__':
    main()
