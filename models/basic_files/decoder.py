from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from . import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from . import rnn_cell
import tensorflow as tf
from . import utils
# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell._linear  # pylint: disable=protected-access

def dynamic_distraction_decoder(config,
                      decoder_inputs,
                      encoder_sentence,
                      initial_state,
                      distract_initial_state,
                      attention_states,
                      query_sentence,
                      attention_states_query,
                      cell,
                      distraction_cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      query_state = None,
                      initial_state_attention=False):
  """RNN decoder with attention for the sequence-to-sequence model.

  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if attention_states.get_shape()[2].value is None:
    raise ValueError("Shape[2] of attention_states must be known: %s"
                     % attention_states.get_shape())

  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(
      scope or "dynamic_distraction_decoder", dtype=dtype) as scope:
    dtype = scope.dtype

    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.

    attn_length_state = attention_states.get_shape()[1].value
    attn_length_query = attention_states_query.get_shape()[1].value

    dim_1 = initial_state.get_shape()[1].value + 2*(encoder_sentence.get_shape()[1].value)
    dim_2 = cell.output_size

    project_initial_state_W = variable_scope.get_variable("Initial_State_W", [dim_1, dim_2])
    project_initial_state_B = variable_scope.get_variable("Initial_State_Bias", [dim_2])

    if attn_length_state is None:
      attn_length_state = attention_states.get_shape()[1]

    if attn_length_query is None:
      attn_length_query = attention_states_query.get_shape()[1]

    attn_size_state = attention_states.get_shape()[2].value
    attn_size_query = attention_states_query.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden_states = array_ops.reshape(
        attention_states, [-1, attn_length_state, 1, attn_size_query])

    hidden_states_query = array_ops.reshape(
        attention_states_query, [-1, attn_length_query, 1, attn_size_query])

    hidden_features_states = []
    hidden_features_query  = []

    v_state = []
    attention_vec_size_state  = attn_size_state  # Size of query vectors for attention.
    
    for a in xrange(num_heads):
      k = variable_scope.get_variable("AttnW_State_%d" % a,
                                      [1, 1, attn_size_state, attention_vec_size_state])

      hidden_features_states.append(nn_ops.conv2d(hidden_states, k, [1, 1, 1, 1], "SAME"))
      
      v_state.append(
          variable_scope.get_variable("AttnV_State_%d" % a, [attention_vec_size_state]))


    v_query = []
    attention_vec_size_query  = attn_size_query  # Size of query vectors for attention.

    for a in xrange(num_heads):
      k = variable_scope.get_variable("AttnW_Query_%d" %a, 
                                      [1, 1, attn_size_query, attention_vec_size_query])

      hidden_features_query.append(nn_ops.conv2d(hidden_states_query, k, [1, 1, 1, 1], "SAME"))
      
      v_query.append(
          variable_scope.get_variable("AttnV_Query_%d" % a, [attention_vec_size_query]))


    init_s = tf.concat([initial_state, encoder_sentence, query_sentence], 1)
    state = math_ops.matmul(init_s, project_initial_state_W) + project_initial_state_B
    distract_state  = [distract_initial_state, distract_initial_state]

    def attention(query):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(query_list,1)
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):
          y = linear(query, attention_vec_size_state, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size_state])
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(
              v_state[a] * math_ops.tanh(hidden_features_states[a] + y), [2, 3])
          a = nn_ops.softmax(s)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length_state, 1, 1]) * hidden_states,
              [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size_state]))
      return ds,a

    def attention_query(query):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(query_list,1)
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_Query_%d" % a):
          y = linear(query, attention_vec_size_query, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size_query])
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(
              v_query[a] * math_ops.tanh(hidden_features_query[a] + y), [2, 3])
          a = nn_ops.softmax(s)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length_query, 1, 1]) * hidden_states_query,
              [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size_query]))
      return ds,a

    outputs = []
    prev = None

    attention_weights_para = []
    attention_weights_query = []
    batch_attn_size_state = array_ops.pack([batch_size, attn_size_state])
    batch_attn_size_query = array_ops.pack([batch_size, attn_size_query])

    attns_state = [array_ops.zeros(batch_attn_size_state, dtype=dtype)
             for _ in xrange(num_heads)]

    attns_query = [array_ops.zeros(batch_attn_size_query, dtype=dtype)
             for _ in xrange(num_heads)]

    for a in attns_state:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size_state])


    for a in attns_query:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size_query])

    if initial_state_attention:
      attns_query, attn_weights_query = attention_query(initial_state)
      list_of_queries = [initial_state, attns_query[0],encoder_sentence]
      attns_state, _ = attention(list_of_queries)

    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()

      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      
      a = attns_state[0]
      if config.config_dir["is_distraction"] == True:
          a = attns_state[0]
          distract_output, distract_state = distraction_cell(a, distract_state)
          x = linear([inp] + [distract_output], input_size, True)
      
      else:
        x = linear([inp] + [a] , input_size, True)
        distract_output = a

      cell_output, state = cell(x, state)
      
      # Run the attention mechanism.

      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                           reuse=True):
          if (config.config_dir["is_query_static"] == True):
            attns_query = query_state
          else:
            attns_query, attns_weight_query = attention_query(state)

          list_of_queries = [state, attns_query[0],encoder_sentence]
          attns_state, attention_weights  = attention(list_of_queries)

      else:
        if (config.config_dir["is_query_static"] == True):
          attns_query = query_state
        else:
          prev_attns_query = attns_query
          attns_query, attns_weight_query= attention_query(state)

        list_of_queries = [state, attns_query[0], encoder_sentence]
        attns_state, attention_weights = attention(list_of_queries)

      with variable_scope.variable_scope("AttnOutputProjection"):

        output = linear([cell_output] + [distract_output], output_size, True)
        #x_shape = variable_scope.get_variable(name = 'x_shape',shape=cell_output.get_shape())
        if loop_function is not None:
          prev = output
        outputs.append(output)
        attention_weights_para.append(attention_weights)
        attention_weights_query.append(attns_weight_query)
        
  return outputs, state, attention_weights_para, attention_weights_query

def dynamic_distraction_decoder_wrapper(config,
                                decoder_inputs,
                                encoder_sentence,
                                initial_state,
                                distract_initial_state,
                                attention_states,
                                query_sentence,
                                attention_states_query,
                                cell_encoder,
                                distraction_cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                embedding_scope = None,
                                dtype=None,
                                scope=None,
                                query_state = None,
                                initial_state_attention=False):
  """RNN decoder with embedding and attention and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_size: Size of the output vectors; if None, use output_size.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    dtype: The dtype to use for the RNN initial states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  if output_size is None:
    output_size = cell_encoder.output_size
  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])


  if embedding_scope == None:
    with variable_scope.variable_scope("dynamic_distraction_decoder_wrapper", dtype=dtype, reuse=None) as s1:
        embedding = variable_scope.get_variable("embedding", [num_symbols, embedding_size])

  else:
    with variable_scope.variable_scope(
        embedding_scope or "dynamic_distraction_decoder_wrapper", dtype=dtype,  reuse = True) as s1:

        embedding = variable_scope.get_variable("embedding",
                                            [num_symbols, embedding_size])
  loop_function = utils._extract_argmax_and_embed(
      embedding, output_projection,
      update_embedding_for_previous) if feed_previous else None
  emb_inp = [
      embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
    
  with variable_scope.variable_scope(
    scope or "dynamic_distraction_decoder_wrapper", dtype =dtype) as scope:
    return dynamic_distraction_decoder(config,
        emb_inp,
        encoder_sentence,
        initial_state=initial_state,
        query_sentence=query_sentence,
        attention_states_query = attention_states_query,
        attention_states=attention_states,
        cell = cell_encoder,
        distract_initial_state = distract_initial_state,
        distraction_cell = distraction_cell,
        query_state = query_state,
        output_size=output_size,
        num_heads=num_heads,
        loop_function=loop_function,
        initial_state_attention=initial_state_attention)


def distraction_decoder_start(config,
                             decoder_inputs,
                             encoder_sentence,
                             attention_states_encoder,
                             query_sentence,
                             attention_states_query,
                             cell_encoder_fw,
                             initial_state,
                             distraction_cell,
                             num_decoder_symbols,
                             embedding_size,
                             initial_embedding = None,
                             num_heads=1,
                             embedding_trainable=False,
                             output_projection=None,
                             feed_previous=False,
                             embedding_scope = None,
                             dtype=None,
                             scope=None,
                             query_state = None,
                             initial_state_attention=False):
    """Embedding sequence-to-sequence model with attention.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an RNN to encode
    embedded encoder_inputs into a state vector. It keeps the outputs of this
    RNN at every step to use for attention later. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs attention decoder, initialized with the last
    encoder state, on embedded decoder_inputs and attending to encoder outputs.

    Args:
      encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      num_encoder_symbols: Integer; number of symbols on the encoder side.
      num_decoder_symbols: Integer; number of symbols on the decoder side.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      num_heads: Number of attention heads that read from attention_states.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_decoder_symbols] and B has
        shape [num_decoder_symbols]; if provided and feed_previous=True, each
        fed previous output will first be multiplied by W and added B.
      feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
        of decoder_inputs will be used (the "GO" symbol), and all other decoder
        inputs will be taken from previous outputs (as in embedding_rnn_decoder).
        If False, decoder_inputs are used as given (the standard decoder case).
      dtype: The dtype of the initial RNN state (default: tf.float32).
      scope: VariableScope for the created subgraph; defaults to
        "embedding_attention_seq2seq".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states.

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x num_decoder_symbols] containing the generated
          outputs.
        state: The state of each decoder cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
    """

    encoder_state = initial_state
    output_size = None
    if output_projection is None:
      cell_encoder_fw = rnn_cell.OutputProjectionWrapper(cell_encoder_fw, num_decoder_symbols)
      output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
      return dynamic_distraction_decoder_wrapper(
          config,
          decoder_inputs,
          encoder_sentence,
          initial_state=encoder_state,
          attention_states=attention_states_encoder,
          query_sentence=query_sentence,
          attention_states_query = attention_states_query,  
          cell_encoder = cell_encoder_fw,
          num_symbols = num_decoder_symbols,
          embedding_size = embedding_size,
          distract_initial_state = encoder_state,
          distraction_cell = distraction_cell,
          query_state = query_state,
          num_heads=num_heads,
          output_size=output_size,
          output_projection=output_projection,
          feed_previous=feed_previous,
          embedding_scope = embedding_scope,
          initial_state_attention=initial_state_attention)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      
      reuse = None if feed_previous_bool else True
      
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse) as scope:

        outputs, state, attention_weights_para, attention_weights_query = dynamic_distraction_decoder_wrapper(
            config,
            decoder_inputs,
            encoder_sentence,
            initial_state=encoder_state,
            attention_states=attention_states_encoder,
            query_sentence = query_sentence,
            attention_states_query = attention_states_query,
            cell_encoder = cell_encoder_fw,
            num_symbols=num_decoder_symbols,
            embedding_size = embedding_size,
            distract_initial_state = encoder_state,
            distraction_cell = distraction_cell,
            query_state = query_state,
            num_heads=num_heads,
            output_size=output_size,
            output_projection=output_projection,
            feed_previous=feed_previous_bool,
            embedding_scope = embedding_scope,
            update_embedding_for_previous=False,
            initial_state_attention=initial_state_attention)

        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)

        #print (len(outputs), outputs[0].get_shape())        
        return outputs + state_list + attention_weights_para + attention_weights_query

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    state_list = outputs_and_state[outputs_len:]
    attention_weights_query = outputs_and_state[-outputs_len:]
    attention_weights_para = outputs_and_state[-2*outputs_len: -1*outputs_len]
    state = state_list[0]
    if nest.is_sequence(encoder_state):
      state = nest.pack_sequence_as(structure=encoder_state,
                                    flat_sequence=state_list)
    return outputs_and_state[:outputs_len], state, attention_weights_para, attention_weights_query

