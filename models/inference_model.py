import tensorflow as tf
import numpy as numpy
from .basic_files.encoder import *
from .basic_files.decoder import *
from .basic_files.rnn_cell import *
from .basic_files.utils import *
import sys



class BasicAttention:

    """ Class Defines the basic attention model : 
        as defined in Paper : A neural attention model for abstractive text summarization
    """ 

    def add_cell(self,hidden_size, cell_type):

        """ Define the rnn_cell to be used in attention model

            Args:
                cell_input: Type of rnn_cell to be used. Default: LSTMCell
                hidden_size : Hidden size of cell
        """

        if(cell_type is "LSTM"):
            return LSTMCell(hidden_size)
        else:
            return GRUCell(hidden_size)

    def add_projectionLayer(self, hidden_size, len_vocab):

        """ Add the projection layer for hidden_size x vocab

            Args:
                hidden_size : The hidden size of the cell
                len_vocab   : The number of symbols in vocabulary
        """
        self.projection_B = tf.get_variable(name="Projection_B", shape=[len_vocab])
        self.projection_W = tf.get_variable(name="Projected_W", shape=[hidden_size, len_vocab])


    def inference(self, config, cell_encoder_fw_type, cell_decoder_type,
    	          encoder_inputs1, decoder_inputs1,
    	          query_inputs, embedding_size, feed_previous,
                  len_vocab, hidden_size, weights, encoder_sequence_length, query_sequence_length, 
                  embedding_trainable, initial_embedding = None, c=None, initial_embedding_encoder=None,
                  initial_embedding_query=None, sequence_indices_encoder=None, sequence_indices_query=None):

        """ Builds the graph for the basic attetion model

            Args:
                encoder_inputs: Placeholder for the encoder sequence
                decoder_inputs: Placeholder for the decoder sequence
                query_inputs  : Placeholder for the query   sequence
                embedding_size: Dimensions of the embedding for encoder and decoder symbols.
                feed_previous : Boolean to decide whether to feed previous state output 
                                to current rnn state for the decoder.
                len_vocab     : Number of symbols in encoder/decoder.
                hidden_size   : Hidden size of the cell state
                c             : The cell that needs to be used.
        
            Returns:
                A list of tensors of size [batch_size * num_symbols], that gives the
                probability distribution over symbols for each time step. The list
                is of size max_sequence_length
        """
        self.add_projectionLayer(hidden_size, len_vocab)

        cell_encoder_fw = self.add_cell(hidden_size, cell_encoder_fw_type)
        cell_encoder_bw = self.add_cell(hidden_size, cell_encoder_fw_type)
        cell_decoder    = self.add_cell(hidden_size, cell_decoder_type)

        if (config.config_dir["is_bidir"]):
        	hs = 2*hidden_size
        else:
        	hs = hidden_size

        if config.config_dir["distraction_cell"] == "LSTM_soft":
        	distract_cell = DistractionLSTMCell_soft(hs, state_is_tuple = True)
        elif config.config_dir["distraction_cell"] == "LSTM_hard":
        	distract_cell = DistractionLSTMCell_hard(hs, state_is_tuple=True)
        elif config.config_dir["distraction_cell"] == "LSTM_sub":
        	distract_cell = DistractionLSTMCell_subtract(hs, state_is_tuple=True)
        elif config.config_dir["distraction_cell"] == "GRU_hard":
        	distract_cell = DistractionGRUCell_hard(hs) 
        elif config.config_dir["distraction_cell"] == "GRU_soft":
        	distract_cell = DistractionGRUCell_soft(hs)
        elif config.config_dir["distraction_decoder_start_cell"] == "GRU_sub":
        	distract_cell = DistractionGRUCell_subtract(hs)

        #enc_cell = DistractionLSTMCell(hidden_size)
        ei = tf.unstack(encoder_inputs1)
        di = tf.unstack(decoder_inputs1)
        qi = tf.unstack(query_inputs)
        encoder_state, encoder_outputs, query_state, query_outputs, embedding_scope, encoder_sentence, query_sentence = encoder(
        										config,
        										encoder_inputs = ei,
                                                query_inputs = qi,
                                                cell_encoder_fw = cell_encoder_fw,
                                                cell_encoder_bw = cell_encoder_bw,
                                                embedding_trainable=embedding_trainable,
                                                sequence_length_encoder = encoder_sequence_length,
                                                sequence_length_query = query_sequence_length,
                                                num_encoder_symbols= len_vocab,
                                                embedding_size = embedding_size,
                                                initial_embedding = initial_embedding,
						initial_embedding_encoder = initial_embedding_encoder,
						initial_embedding_query = initial_embedding_query,
                                                dtype=tf.float32, 
                                                sequence_indices_encoder = tf.unstack(sequence_indices_encoder), 
                                                sequence_indices_query = tf.unstack(sequence_indices_query))

        if (config.config_dir["diff_vocab"]):
        	embedding_scope = None

        outputs, state, attention_weights_para, attention_weights_query = distraction_decoder_start(config,
												decoder_inputs = di,
                                                  encoder_sentence = encoder_sentence,
												attention_states_encoder = encoder_outputs, 
						  query_sentence = query_sentence,						
                                                   attention_states_query = query_outputs,
												initial_state = encoder_state, 
                                                cell_encoder_fw = cell_decoder,
                                                distraction_cell = distract_cell,
                                                embedding_trainable=embedding_trainable,
                                                num_decoder_symbols= len_vocab,
                                                embedding_size = embedding_size,
                                                output_projection= (self.projection_W, self.projection_B),
                                                feed_previous= feed_previous,
                                                initial_embedding = initial_embedding,
                                                query_state = query_state,
                                                embedding_scope = embedding_scope,
                                                dtype=tf.float32)

        self.final_outputs = [tf.matmul(o, self.projection_W) + self.projection_B for o in outputs]

        # Convert the score to a probability distribution.
        #self.final_outputs = [tf.nn.softmax(o) for o in self.final_outputs]

        return self.final_outputs, attention_weights_para, attention_weights_query


    def loss_op(self, outputs, labels, weights):

        """ Calculate the loss from the predicted outputs and the labels

            Args:
                outputs : A list of tensors of size [batch_size * num_symbols]
                labels : A list of tensors of size [sequence_length * batch_size]

            Returns:
                loss: loss of type float
        """

        _labels = tf.unstack(labels)
        all_ones       = [tf.ones(shape=tf.shape(_labels[0])) for _ in range(len(_labels))]
        weights = tf.to_float(weights)
        _weights = tf.unstack(weights)
        #print(_weights[0].get_shape())
        loss_per_batch = sequence_loss(outputs, _labels, _weights)

        self.calculated_loss =  loss_per_batch
        return loss_per_batch


    def training(self, loss, learning_rate):

        """ Creates an optimizer and applies the gradients to all trainable variables.

            Args:
                loss : Loss value passed from function loss_op
                learning_rate : Learning rate for GD.

            Returns:
                train_op : Optimizer for training
        """

        optimizer = tf.train.AdamOptimizer(learning_rate)
        #train_op = optimizer.minimize(loss)
        grad = optimizer.compute_gradients(loss)
        grad = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grad ]
	train_op = optimizer.apply_gradients(grad)
	return train_op


# To test the model
def main():
    n = basic_attention_model(c)
    n.inference(int(100))
    print ("Inference")
    l = n.loss_op(n.final_outputs, int(100))
    print ("Loss")
    n.training(l)
    print ("Train")

if __name__ == '__main__':
    main()
