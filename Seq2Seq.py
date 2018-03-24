
# coding: utf-8

# In[1]:


import sys, os
import csv, re, copy, random
import pickle
import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt
from tensorlayer.layers import *
from sklearn.utils import shuffle
from time import gmtime, strftime
import numpy as np
import time
import logging
from collections import Counter
from gensim.models import KeyedVectors
from pycorenlp import StanfordCoreNLP
from Dictionary import *
from nltk.tree import Tree
nlp = StanfordCoreNLP('http://localhost:9000')


# In[2]:


def loadSentences(filename):
    f = open(filename, encoding="utf8")
    reader = csv.DictReader(f, delimiter='\t')
    sentences = [row for row in reader]
    return sentences


# In[3]:


def loadAllTextSentences():
    sentences = loadSentences('dataset/Training.sentence')
    sentences.extend(loadSentences('dataset/SampleSet.sentence'))
    sentences.extend(loadSentences('dataset/Task1NeuV3_corrected.sentence'))
    # print(len(sentences))
    TextSentenceID = dict()
    vocabulary = set()
    for line in sentences:
        id = line['Sentence-ID'][4:]
        text = line['Sentence']
        output = nlp.annotate(text, properties={'annotators': 'tokenize', 'outputFormat': 'json'})
        if id not in TextSentenceID:
            TextSentenceID[id] = {'id': id,
                                  'text': text,
                                  'pmid': line['PMID'],
                                  'tokens': [i['originalText'] for i in output['tokens']]}
            vocabulary = vocabulary.union(set(TextSentenceID[id]['tokens']))
    #     else:
    #         assert (TextSentenceID[id]['text'] == line['Sentence']), 'ID: %s \n Text1: %s \n Text2: %s'%(id, TextSentenceID[id]['text'], line['Sentence']) 
    #         assert (TextSentenceID[id]['pmid'] == line['PMID']), 'ID: %s \n PMID1: %s \n PMID2: %s'%(id, TextSentenceID[id]['pmid'], line['PMID'])
    print('Downloaded text sentences:', len(TextSentenceID), 'sentences')
    print('Total vocabulary:', len(vocabulary), 'words')
    return TextSentenceID, vocabulary


# In[4]:


def tokeniseBEL(bs, mapToHGNC = False):
    bs = re.subn(r',GOCCID:\d+', '', bs) # Replace GOCCID (additional parameters of tloc)
    bs = re.subn(r',sub\([\w,]*?\)', '', bs[0]) # Remove all sub(_,_,_)
    bs = re.subn(r',trunc\(\d+\)', '', bs[0]) # Remove all trunc(_)
    terms = re.findall(r'(a|bp|path|g|m|r|p)\(([A-Z]+)\:([^"]*?|".*?")(,pmod\(.*?\))?\)', bs[0])
    # print(terms)
    bs = re.subn(r'(a|bp|path|g|m|r|p)\(([A-Z]+)\:([^"]*?|".*?")(,pmod\(.*?\))?\)', '@', bs[0])
    assert bs[1] == len(terms)
    # print(bs)
    relations = re.findall(r'\s((?:->)|(?:-\|)|(?:increases)|(?:decreases)|(?:directlyIncreases)|(?:directlyDecreases))\s', bs[0])
    # print(relations)
    bs = re.subn(r'\s((?:->)|(?:-\|)|(?:increases)|(?:decreases)|(?:directlyIncreases)|(?:directlyDecreases))\s', '&', bs[0])
    assert bs[1] == len(relations)
    # print(bs)
    functions = re.findall(r'((?:act)|(?:complex)|(?:tloc)|(?:deg)|(?:kin)|(?:tscript)|(?:cat)|(?:sec)|(?:chap)|(?:gtp)|(?:pep)|(?:phos)|(?:ribo)|(?:tport)|(?:surf))', bs[0])
    # print(functions)
    bs = re.subn(r'((?:act)|(?:complex)|(?:tloc)|(?:deg)|(?:kin)|(?:tscript)|(?:cat)|(?:sec)|(?:chap)|(?:gtp)|(?:pep)|(?:phos)|(?:ribo)|(?:tport)|(?:surf))', '$', bs[0])
    assert bs[1] == len(functions)
    # print(bs[0]) # Term = @, Function = $, Relation = &
    bs = re.subn(r', ', ',', bs[0]) # Remove white space after comma
    template = bs[0]
    return stringToTokens(template, terms, relations, functions, mapToHGNC)


# In[5]:


def stringToTokens(template, terms, relations, functions, mapToHGNC = False):
    aList = []
    typeDict = {'p':'p', 'bp':'bp', 'path':'path', 'a':'a', 'g':'p', 'm':'p', 'r':'p'}
    relationDict = {'->': ' -> ', '-|': ' -| ', 'increases':' -> ', 'decreases':' -| ', 'directlyIncreases':' -> ', 'directlyDecreases':' -| '}
    functionDict = {'act':'act', 'kin':'act', 'tscript':'act', 'cat':'act', 'chap':'act', 'gtp':'act', 'pep':'act', 'phos':'act', 'ribo':'act', 'tport':'act',
                    'complex':'complex', 
                    'tloc': 'tloc', 'sec':'tloc', 'surf':'tloc', 
                    'deg': 'deg'}
    for s in template:
        if s == '@': # Term
            termTuple = terms.pop(0)
            ns = termTuple[1]
            symbol = termTuple[2]
            if mapToHGNC:
                if ns == 'EGID': # Transform namespace (EGID to HGNC)
                    if symbol in EGID2HGNC:
                        ns = 'HGNC'
                        symbol = EGID2HGNC[symbol]
                        if any([a in symbol for a in [' ', '(', ')', '+', '-']]):
                            symbol = '"' + symbol + '"'
                    else:
                        return False
                elif ns == 'MGI': # Transform namespace (MGI to HGNC)
                    if symbol in MGI2HGNC:
                        ns = 'HGNC'
                        symbol = MGI2HGNC[symbol]
                        if any([a in symbol for a in [' ', '(', ')', '+', '-']]):
                            symbol = '"' + symbol + '"'
                    else:
                        return False
            if termTuple[3] == '':
                aList.extend([typeDict[termTuple[0]], '(', ns+':'+symbol, ')'])
            else:
                aList.extend([typeDict[termTuple[0]], '(', ns+':'+symbol, ',', 'pmod(P)', ')'])
        elif s == '&': # Relation
            aList.append(relationDict[relations.pop(0)])
        elif s == '$': # Function
            aList.append(functionDict[functions.pop(0)])
        else: # brackets and comma
            aList.append(s)
    return aList


# In[6]:


def averageWordVectors(sentence, word_vectors):
    output = nlp.annotate(sentence, properties={'annotators': 'tokenize', 'outputFormat': 'json'})
    sentence_tokens = [i['originalText'] for i in output['tokens']]
    sentence_matrix = np.array([word_vectors[tok] for tok in sentence_tokens if tok in word_vectors.vocab])
#     print(len(sentence_matrix))
    if len(sentence_matrix) > 0:
        average_sentence_vector = np.mean(sentence_matrix, axis = 0)
    else:
        average_sentence_vector = np.random.uniform(-0.1, 0.1, emb_dim)
    return average_sentence_vector    


# In[7]:


def words2index(listOfWords, w2idx):
    lst = []
    for w in listOfWords:
        if w in w2idx:
            lst.append(w2idx[w])
        else:
            lst.append(w2idx['unk'])
#             print(listOfWords)
            if ' -> ' in w2idx:
                print('Word in BEL converted to unknown:', w)
            else:
                print('Word in Text converted to unknown:', w)
    return lst


# In[8]:


def validate():
    numTest = 30
    with g.as_default() as graph:
        sumLoss = []
        i = 1
        random.shuffle(sampleZipSort)
        validation_data = sampleZipSort[:numTest]
        for enc_seq, tar_seq in validation_data:
            x = enc_seq[::-1]
            sentence, logits = predict(x, val_network_encode, val_network_decode, net_val, encode_seqs3, decode_seqs3)
            logits = logits.reshape((logits.shape[0]*logits.shape[1], logits.shape[2]))
            print(''.join(sentence), logits.shape)
            tar_seq = tar_seq + [end_id]
            
            if len(logits) > len(tar_seq):
                tar_seq = tar_seq + [end_id]*(len(logits) - len(tar_seq))
            elif len(logits) < len(tar_seq):
                end_logit = np.zeros((len(tar_seq)-len(logits), logits.shape[1]), dtype=np.float32)
                end_logit[:, end_id:end_id+1] = 1
                logits = np.concatenate((logits, end_logit), axis = 0) 
            
#             print(len(tar_seq), logits.shape)
            tar_mark = [1]*len(tar_seq)

            val_loss = tl.cost.cross_entropy_seq_with_mask(logits=logits, target_seqs=target_seqs3, input_mask=target_mask3, return_details=False, name='cost_val')
            lossX = sess.run([val_loss], {target_seqs3: [tar_seq], target_mask3: [tar_mark]})
            print('Error', i, lossX)
            sumLoss += lossX
            i += 1
        return sum(sumLoss)/len(sumLoss)


# In[9]:


def predict(enc_seq, encode_layer, decode_layer, output_layer, encode_seqs, decode_seqs, needLogits = True, needAttention = False):
    state, memory = sess.run([encode_layer.final_state, encode_layer.outputs], 
                             {encode_seqs: [enc_seq]})
    feed_dict = {decode_layer.initial_state.cell_state:state,
                 decode_layer.memory:memory,
                 decode_seqs: [[start_id]]}
    o, state = sess.run([tf.nn.softmax(output_layer.outputs), decode_layer.final_state], feed_dict)
    
    w_id = tl.nlp.sample_top(o[0], top_k=1)
    w = B_idx2w[w_id]
    sentence = [w]
    logits = [o]
    attention = state.alignments
    for _ in range(30): # max sentence length
        o, state = sess.run([tf.nn.softmax(output_layer.outputs), decode_layer.final_state],
                        {decode_layer.initial_state:state,
                         decode_layer.memory:memory,
                         decode_seqs: [[w_id]]})
        attention = np.concatenate((attention, state.alignments), axis = 0)
        if needLogits:
            logits.append(o)
        w_id = tl.nlp.sample_top(o[0], top_k=1)
        w = B_idx2w[w_id] 
        if w_id == end_id:
#             print(attention)
#             print(attention.shape)
            break
        sentence = sentence + [w]
    if needLogits and needAttention:
        return sentence, np.array(logits), attention
    elif needLogits:
        return sentence, np.array(logits)
    elif needAttention:
        return sentence, attention
    else:
        return sentence


# In[10]:


def model(encode_seqs, decode_seqs, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        # for translation, you need 2 seperated embedding layers
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                inputs = encode_seqs,
                vocabulary_size = T_vocab_size_total,
                embedding_size = emb_dim,
                name = 'encode_seq_embedding')
            net_decode = EmbeddingInputlayer(
                inputs = decode_seqs,
                vocabulary_size = B_vocab_size_total,
                embedding_size = emb_dim,
                name = 'decode_seq_embedding')
            vs.reuse_variables()
            tl.layers.set_name_reuse(True)
        net_rnn = Seq2Seq(net_encode, net_decode,
                cell_fn = tf.contrib.rnn.BasicLSTMCell,
                n_hidden = emb_dim,
                initializer = tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
                decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
                initial_state_encode = None,
                dropout = (0.5 if is_train else None),
                n_layer = 1,
                return_seq_2d = True,
                name = 'seq2seq')
        net_out = DenseLayer(net_rnn, n_units=B_vocab_size_total, act=tf.identity, name='output')
    return net_out, net_rnn, net_encode


# In[11]:


def modelWithAttention(encode_seqs, decode_seqs, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        # for translation, you need 2 seperated embedding layers
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                inputs = encode_seqs,
                vocabulary_size = T_vocab_size_total,
                embedding_size = emb_dim,
                name = 'encode_seq_embedding')
            net_decode = EmbeddingInputlayer(
                inputs = decode_seqs,
                vocabulary_size = B_vocab_size_total,
                embedding_size = emb_dim,
                name = 'decode_seq_embedding')
            vs.reuse_variables()
            tl.layers.set_name_reuse(True)
        network_encode = DynamicRNNLayer(
            net_encode,
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            n_hidden=emb_dim,
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            initial_state=None,
            dropout=(0.5 if is_train else None),
            n_layer=1,
            sequence_length=retrieve_seq_length_op2(encode_seqs),
            return_last=False,
            return_seq_2d=False,
            name='seq2seq_encode')
        network_decode = DynamicRNNLayerWithAttention(
            net_decode,
            cell_fn=tf.contrib.rnn.BasicLSTMCell,
            attention_mechanism_fn=tf.contrib.seq2seq.LuongAttention,
            memory=network_encode.outputs,
            n_hidden=emb_dim,
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            initial_state=network_encode.final_state,
           # initial_state=(network_encode.final_state if is_train else None),
            dropout=(0.5 if is_train else None),
            n_layer=1,
            sequence_length=retrieve_seq_length_op2(decode_seqs),
            return_last=False,
            return_seq_2d=True,
            name='seq2seq_decode')
        net_out = DenseLayer(network_decode, n_units=B_vocab_size_total, act=tf.identity, name='output')
    return net_out, network_encode, net_encode, network_decode, network_decode.cell, net_decode


# In[12]:


class DynamicRNNLayerWithAttention(Layer):
    """
    The :class:`DynamicRNNLayer` class is a dynamic recurrent layer, see ``tf.nn.dynamic_rnn``.

    Parameters
    ----------
    layer : :class:`Layer`
        Previous layer
    cell_fn : TensorFlow cell function
        A TensorFlow core RNN cell
            - See `RNN Cells in TensorFlow <https://www.tensorflow.org/api_docs/python/>`__
            - Note TF1.0+ and TF1.0- are different
    cell_init_args : dictionary or None
        The arguments for the cell function.
    n_hidden : int
        The number of hidden units in the layer.
    initializer : initializer
        The initializer for initializing the parameters.
    sequence_length : tensor, array or None
        The sequence length of each row of input data, see ``Advanced Ops for Dynamic RNN``.
            - If None, it uses ``retrieve_seq_length_op`` to compute the sequence length, i.e. when the features of padding (on right hand side) are all zeros.
            - If using word embedding, you may need to compute the sequence length from the ID array (the integer features before word embedding) by using ``retrieve_seq_length_op2`` or ``retrieve_seq_length_op``.
            - You can also input an numpy array.
            - More details about TensorFlow dynamic RNN in `Wild-ML Blog <http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/>`__.
    initial_state : None or RNN State
        If None, `initial_state` is zero state.
    dropout : tuple of float or int
        The input and output keep probability (input_keep_prob, output_keep_prob).
            - If one int, input and output keep probability are the same.
    n_layer : int
        The number of RNN layers, default is 1.
    return_last : boolean or None
        Whether return last output or all outputs in each step.
            - If True, return the last output, "Sequence input and single output"
            - If False, return all outputs, "Synced sequence input and output"
            - In other word, if you want to stack more RNNs on this layer, set to False.
    return_seq_2d : boolean
        Only consider this argument when `return_last` is `False`
            - If True, return 2D Tensor [n_example, n_hidden], for stacking DenseLayer after it.
            - If False, return 3D Tensor [n_example/n_steps, n_steps, n_hidden], for stacking multiple RNN after it.
    dynamic_rnn_init_args : dictionary
        The arguments for ``tf.nn.dynamic_rnn``.
    name : str
        A unique layer name.

    Attributes
    ------------
    outputs : tensor
        The output of this layer.

    final_state : tensor or StateTuple
        The finial state of this layer.
            - When `state_is_tuple` is `False`, it is the final hidden and cell states, `states.get_shape() = [?, 2 * n_hidden]`.
            - When `state_is_tuple` is `True`, it stores two elements: `(c, h)`.
            - In practice, you can get the final state after each iteration during training, then feed it to the initial state of next iteration.

    initial_state : tensor or StateTuple
        The initial state of this layer.
            - In practice, you can set your state at the begining of each epoch or iteration according to your training procedure.

    batch_size : int or tensor
        It is an integer, if it is able to compute the `batch_size`; otherwise, tensor for dynamic batch size.

    sequence_length : a tensor or array
        The sequence lengths computed by Advanced Opt or the given sequence lengths, [batch_size]

    Notes
    -----
    Input dimension should be rank 3 : [batch_size, n_steps(max), n_features], if no, please see :class:`ReshapeLayer`.

    Examples
    --------
    Synced sequence input and output, for loss function see ``tl.cost.cross_entropy_seq_with_mask``.

    >>> input_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="input")
    >>> net = tl.layers.EmbeddingInputlayer(
    ...             inputs = input_seqs,
    ...             vocabulary_size = vocab_size,
    ...             embedding_size = embedding_size,
    ...             name = 'seq_embedding')
    >>> net = tl.layers.DynamicRNNLayer(net,
    ...             cell_fn = tf.contrib.rnn.BasicLSTMCell, # for TF0.2 use tf.nn.rnn_cell.BasicLSTMCell,
    ...             n_hidden = embedding_size,
    ...             dropout = (0.7 if is_train else None),
    ...             sequence_length = tl.layers.retrieve_seq_length_op2(input_seqs),
    ...             return_seq_2d = True,                   # stack denselayer or compute cost after it
    ...             name = 'dynamicrnn')
    ... net = tl.layers.DenseLayer(net, n_units=vocab_size, name="output")

    References
    ----------
    - `Wild-ML Blog <http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/>`__
    - `dynamic_rnn.ipynb <https://github.com/dennybritz/tf-rnn/blob/master/dynamic_rnn.ipynb>`__
    - `tf.nn.dynamic_rnn <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard8/tf.nn.dynamic_rnn.md>`__
    - `tflearn rnn <https://github.com/tflearn/tflearn/blob/master/tflearn/layers/recurrent.py>`__
    - ``tutorial_dynamic_rnn.py``

    """

    def __init__(
            self,
            layer,
            cell_fn,  #tf.nn.rnn_cell.LSTMCell,
            attention_mechanism_fn,
            memory,
            cell_init_args=None,
            n_hidden=256,
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            sequence_length=None,
            initial_state=None,
            dropout=None,
            n_layer=1,
            return_last=None,
            return_seq_2d=False,
            dynamic_rnn_init_args=None,
            name='dyrnn',
    ):
#         self.initial_state_from_encoder = initial_state
        
        if dynamic_rnn_init_args is None:
            dynamic_rnn_init_args = {}
        if cell_init_args is None:
            cell_init_args = {'state_is_tuple': True}
        if return_last is None:
            return_last = True

        Layer.__init__(self, name=name)
        if cell_fn is None:
            raise Exception("Please put in cell_fn")
        if 'GRU' in cell_fn.__name__:
            try:
                cell_init_args.pop('state_is_tuple')
            except Exception:
                logging.warning("pop state_is_tuple fails.")
        self.inputs = layer.outputs
        self.memory = memory
        print("  [TL] DynamicRNNLayerWithAttention %s: n_hidden:%d, in_dim:%d in_shape:%s cell_fn:%s dropout:%s n_layer:%d" %
                     (self.name, n_hidden, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__, dropout, n_layer))

        logging.info("DynamicRNNLayerWithAttention %s: n_hidden:%d, in_dim:%d in_shape:%s cell_fn:%s dropout:%s n_layer:%d" %
                     (self.name, n_hidden, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__, dropout, n_layer))

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            self.inputs.get_shape().with_rank(3)
        except Exception:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps(max), n_features]")

        # Get the batch_size
        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]
        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            logging.info("       batch_size (concurrent processes): %d" % batch_size)
        else:
            from tensorflow.python.ops import array_ops
            batch_size = array_ops.shape(self.inputs)[0]
            logging.info("       non specified batch_size, uses a tensor instead.")
        self.batch_size = batch_size

        # Creats the cell function
        # cell_instance_fn=lambda: cell_fn(num_units=n_hidden, **cell_init_args) # HanSheng
        rnn_creator = lambda: cell_fn(num_units=n_hidden, **cell_init_args)
        
        # ============================ PLJ added attention mechanism ====================================
        attention_mechanism = attention_mechanism_fn(num_units = n_hidden,
                                memory = self.memory,
                                memory_sequence_length = None # Might be made more accurate later 
                                )
        attn_cell_creator = lambda: tf.contrib.seq2seq.AttentionWrapper(rnn_creator(), attention_mechanism)
        # ===============================================================================================            
        # Apply dropout
        if dropout:
            if isinstance(dropout, (tuple, list)):
                in_keep_prob = dropout[0]
                out_keep_prob = dropout[1]
            elif isinstance(dropout, float):
                in_keep_prob, out_keep_prob = dropout, dropout
            else:
                raise Exception("Invalid dropout type (must be a 2-D tuple of " "float)")
            try:  # TF1.0
                DropoutWrapper_fn = tf.contrib.rnn.DropoutWrapper
            except Exception:
                DropoutWrapper_fn = tf.nn.rnn_cell.DropoutWrapper

            # cell_instance_fn1=cell_instance_fn        # HanSheng
            # cell_instance_fn=DropoutWrapper_fn(
            #                     cell_instance_fn1(),
            #                     input_keep_prob=in_keep_prob,
            #                     output_keep_prob=out_keep_prob)
            cell_creator = lambda is_last=True:                     DropoutWrapper_fn(attn_cell_creator(),
                                      input_keep_prob=in_keep_prob,
                                      output_keep_prob=out_keep_prob if is_last else 1.0)
        else:
            cell_creator = attn_cell_creator
        self.cell = cell_creator()
        # Apply multiple layers
        if n_layer > 1:
            try:
                MultiRNNCell_fn = tf.contrib.rnn.MultiRNNCell
            except Exception:
                MultiRNNCell_fn = tf.nn.rnn_cell.MultiRNNCell

            # cell_instance_fn2=cell_instance_fn # HanSheng
            try:
                # cell_instance_fn=lambda: MultiRNNCell_fn([cell_instance_fn2() for _ in range(n_layer)], state_is_tuple=True) # HanSheng
                self.cell = MultiRNNCell_fn([cell_creator(is_last=i == n_layer - 1) for i in range(n_layer)], state_is_tuple=True)
            except Exception:  # when GRU
                # cell_instance_fn=lambda: MultiRNNCell_fn([cell_instance_fn2() for _ in range(n_layer)]) # HanSheng
                self.cell = MultiRNNCell_fn([cell_creator(is_last=i == n_layer - 1) for i in range(n_layer)])

        # self.cell=cell_instance_fn() # HanSheng

        # Initialize initial_state
        if initial_state is None:
            self.initial_state = self.cell.zero_state(batch_size, dtype=D_TYPE)  # dtype=tf.float32)
        else:
            try:
                self.initial_state = self.cell.zero_state(batch_size, dtype=D_TYPE).clone(cell_state=initial_state) 
            except AttributeError:
                self.initial_state = initial_state

        # Computes sequence_length
        if sequence_length is None:
            try:  # TF1.0
                sequence_length = retrieve_seq_length_op(self.inputs if isinstance(self.inputs, tf.Tensor) else tf.stack(self.inputs))
            except Exception:  # TF0.12
                sequence_length = retrieve_seq_length_op(self.inputs if isinstance(self.inputs, tf.Tensor) else tf.pack(self.inputs))
        
        # Main - Computes outputs and last_states
#         with tf.variable_scope(tf.get_variable_scope()) as scope:
        with tf.variable_scope(name, initializer=initializer) as vs:
            outputs, last_states = tf.nn.dynamic_rnn(
                cell=self.cell,
                # inputs=X
                inputs=self.inputs,
                dtype=tf.float32,
                sequence_length=sequence_length,
                initial_state=self.initial_state,
                **dynamic_rnn_init_args)
            rnn_variables = tf.get_collection(TF_GRAPHKEYS_VARIABLES, scope=vs.name)

            # logging.info("     n_params : %d" % (len(rnn_variables)))
            # Manage the outputs
            if return_last:
                # [batch_size, n_hidden]
                # outputs = tf.transpose(tf.pack(outputs), [1, 0, 2]) # TF1.0 tf.pack --> tf.stack
                self.outputs = advanced_indexing_op(outputs, sequence_length)
            else:
                # [batch_size, n_step(max), n_hidden]
                # self.outputs = result[0]["outputs"]
                # self.outputs = outputs    # it is 3d, but it is a list
                if return_seq_2d:
                    # PTB tutorial:
                    # 2D Tensor [n_example, n_hidden]
                    try:  # TF1.0
                        self.outputs = tf.reshape(tf.concat(outputs, 1), [-1, n_hidden])
                    except Exception:  # TF0.12
                        self.outputs = tf.reshape(tf.concat(1, outputs), [-1, n_hidden])
                else:
                    # <akara>:
                    # 3D Tensor [batch_size, n_steps(max), n_hidden]
                    max_length = tf.shape(outputs)[1]
                    batch_size = tf.shape(outputs)[0]

                    try:  # TF1.0
                        self.outputs = tf.reshape(tf.concat(outputs, 1), [batch_size, max_length, n_hidden])
                    except Exception:  # TF0.12
                        self.outputs = tf.reshape(tf.concat(1, outputs), [batch_size, max_length, n_hidden])
                    # self.outputs = tf.reshape(tf.concat(1, outputs), [-1, max_length, n_hidden])
#             tf.get_variable_scope().reuse_variables()

        # Final state
        self.final_state = last_states
        
        self.sequence_length = sequence_length

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend([self.outputs])
        self.all_params.extend(rnn_variables)


# In[13]:


def createTextDict():
    T_idx2w = ['_', 'unk'] + list(vocabulary) 
    for word in word_vectors.vocab.keys():
        if word not in T_idx2w:
            T_idx2w.append(word)
        if len(T_idx2w) >= T_vocab_size + 2:
            break
    T_idx2w.extend(['start_id', 'end_id'])
    T_w2idx = dict([(T_idx2w[i], i) for i in range(len(T_idx2w))])
    T_vocab_size_total = len(T_idx2w)
    print('Finish creating text dict (Total text vocab size = %d)'%T_vocab_size_total)
    return T_idx2w, T_w2idx, T_vocab_size_total


# In[14]:


def createBELTokenDict():
    B_idx2w = []
    B_idx2w.extend(generalTokens)
    B_idx2w.extend(['HGNC:'+x if all([a not in x for a in [' ', '(', ')', '+', '-']]) else 'HGNC:"'+x+'"' for x in HGNCDict.keys()])
    B_idx2w.extend(['CHEBI:'+x if all([a not in x for a in [' ', '(', ')', '+', '-']]) else 'CHEBI:"'+x+'"' for x in ChEBIDict.keys()])
    B_idx2w.extend(['GOBP:'+x if all([a not in x for a in [' ', '(', ')', '+', '-']]) else 'GOBP:"'+x+'"' for x in GOBPDict.keys()])
    B_idx2w.extend(['MESHD:'+x if all([a not in x for a in [' ', '(', ')', '+', '-']]) else 'MESHD:"'+x+'"' for x in MESHDict.keys()])
    B_idx2w.extend(['start_id', 'end_id'])
    B_w2idx = dict([(B_idx2w[i], i) for i in range(len(B_idx2w))])
    B_vocab_size_total = len(B_idx2w)
    print('Finish creating BEL token dict (Total text vocab size = %d)'%B_vocab_size_total)
    return B_idx2w, B_w2idx, B_vocab_size_total


# In[15]:


def loadWordEmbeddings():
    if os.path.exists("word_embedding.pickle"):
        print('Downloading word embeddings from a file')
        word_embedding = pickle.load(open("word_embedding.pickle", "rb"))
    else:
        print('Cannot find the word embeddings file, so creating the embeddings from scratch.')
        word_embedding = np.random.uniform(-0.1, 0.1, (2, emb_dim))
        count = 2
        for i in range(2, T_vocab_size_total-2):
            if i%100 == 0:
                print(i)
            if T_idx2w[i] in word_vectors.vocab:
        #         print(word_embedding.shape, np.array([word_vectors[T_idx2w[i]]]).shape)
                word_embedding = np.append(word_embedding, [word_vectors[T_idx2w[i]]], axis = 0)
            else:
                word_embedding = np.append(word_embedding, [np.random.uniform(-0.1, 0.1, emb_dim)], axis = 0)
                count += 1
        word_embedding = np.append(word_embedding, [np.random.uniform(-0.1, 0.1, emb_dim)], axis = 0)
        word_embedding = np.append(word_embedding, [np.random.uniform(-0.1, 0.1, emb_dim)], axis = 0)
        print(count+2)
        pickle.dump(word_embedding, open("word_embedding.pickle", "wb"))
    print('Finish loading word embeddings (shape: %s)' % (str(word_embedding.shape)))
    return word_embedding


# In[16]:


def loadBELTokenEmbeddings():
    if os.path.exists("BELToken_embedding.pickle"):
        print('Downloading BELToken embeddings from a file')
        BELToken_embedding = pickle.load(open("BELToken_embedding.pickle", "rb"))
    else:
        print('Cannot find the BELToken embeddings file, so creating the embeddings from scratch.')
        BELToken_embedding = np.random.uniform(-0.1, 0.1, (len(generalTokens), emb_dim))
        print(BELToken_embedding.shape)
        for x in HGNCDict.keys():
            if x in word_vectors.vocab:
                BELToken_embedding = np.append(BELToken_embedding, [word_vectors[x]], axis = 0)
            else:
                BELToken_embedding = np.append(BELToken_embedding, [np.random.uniform(-0.1, 0.1, emb_dim)], axis = 0)
        print(BELToken_embedding.shape)
        for d in [ChEBIDict, GOBPDict, MESHDict]:
            for x in d.keys():
                BELToken_embedding = np.append(BELToken_embedding, [averageWordVectors(d[x]['definition'], word_vectors)], axis = 0)
            print(BELToken_embedding.shape)
        BELToken_embedding = np.append(BELToken_embedding, [np.random.uniform(-0.1, 0.1, emb_dim)], axis = 0)
        BELToken_embedding = np.append(BELToken_embedding, [np.random.uniform(-0.1, 0.1, emb_dim)], axis = 0)
        print(BELToken_embedding.shape)
        assert BELToken_embedding.shape[0] == len(B_idx2w)
        pickle.dump(BELToken_embedding, open("BELToken_embedding.pickle", "wb"))
    print('Finish loading BELToken embeddings (shape: %s)' % (str(BELToken_embedding.shape)))
    return BELToken_embedding


# In[17]:


def loadTrainingData(filename):    
    sentences = loadSentences(filename)
    trainBSentences = []
    trainBTokenized = []
    trainTSentences = []
    trainTTokenized = []
    for line in sentences:
        BELTokens = tokeniseBEL(line['BEL-normalised'], mapToHGNC = True)
        if not BELTokens:
            print('Deleted BEL sentence:', line['BEL-normalised'])
            continue
        trainBSentences.append(line['BEL-normalised'])
        trainBTokenized.append(BELTokens)
        trainTSentences.append(TextSentenceID[line['Sentence-ID'][4:]]['text'])
        trainTTokenized.append(TextSentenceID[line['Sentence-ID'][4:]]['tokens'])
    trainT1 = [words2index(sublist, T_w2idx) for sublist in trainTTokenized]
    trainB1 = [words2index(sublist, B_w2idx) for sublist in trainBTokenized]
    assert len(trainT1) == len(trainB1)
    zipsort = sorted(zip(trainT1,trainB1), key=lambda pair: len(pair[0]))
    trainT = [x for x, y in zipsort]
    trainB = [y for x, y in zipsort] # Sort to make a batch have sentences with similar lengths
#     print(trainB[150], trainT[150])
    return trainT, trainB, zipsort


# In[18]:


def testModel(filename):
    sentences = loadSentences(filename) # filename = 'dataset/Task1NeuV3_corrected.sentence' or 'dataset/SampleSet.sentence'
    realFilename = filename.split('/')[1].split('.')[0]
    sampleTSentences = [TextSentenceID[line['Sentence-ID'][4:]]['text'] for line in sentences]
    sampleTTokenized = [TextSentenceID[line['Sentence-ID'][4:]]['tokens'] for line in sentences]
    sampleT = [words2index(sublist, T_w2idx) for sublist in sampleTTokenized]
    f = open('results/' + realFilename + '_' + strftime("%Y%m%d%H%M%S", gmtime()) +'.txt', 'w')
    for i in range(len(sampleT)):
        sentence_id = sentences[i]['Sentence-ID'][4:]
        seed_id = sampleT[i]
        seed_id = seed_id[::-1]
        sentence = predict(seed_id, net_rnn, network_decode, net, encode_seqs2, decode_seqs2, needLogits = False)
        print(i, ''.join(sentence))
        f.write(sentence_id + '\t' + ''.join(sentence) + '\n')
    f.close()


# In[19]:


def testInference(visualiseAttention = False):
    seeds = ["MMP-2 gelatinolytic activity was higher in cells infected with PBS (mock) and Ad-SV.", 
             # p(HGNC:MMP2) increases act(p(HGNC:MMP2))
            "In contrast, vandetanib treatment induced a 2.3-fold increase in eNOS mRNA in B16.F10 vasculature.", 
             # kin(p(MGI:Kdr)) decreases p(MGI:Nos3)
            "Thus, extramitochondrially targeted AIF is a dominant cell death inducer.", 
             # p(MGI:Aifm1) increases bp(GOBP:"cell death")
            "Binding of PIAS1 to human AR DNA+ligand binding domains was androgen dependent in the yeast liquid beta-galactosidase assay.", 
             # a(CHEBI:androgen) -> complex(p(HGNC:AR),p(HGNC:PIAS1))"
            "The data suggest that genistein may inhibit CFTR by two mechanisms.", 
             # a(CHEBI:genistein) decreases p(MGI:Cftr)
            "LPS-induced NO synthesis feedback regulates itself through up-regulation of OPN promoter activity and gene transcription." 
             # a(CHEBI:lipopolysaccharide) increases a(CHEBI:"nitric oxide"), p(MGI:Spp1) decreases a(CHEBI:"nitric oxide") 
            ] 
    for seed in seeds:
        print("Input >", seed)
        output = nlp.annotate(seed, properties={'annotators': 'tokenize', 'outputFormat': 'json'})
        seed_tokens = [i['originalText'] for i in output['tokens']]
        # print(seed_tokens)
        seed_id = [T_w2idx[w] if w in T_w2idx else T_w2idx['unk'] for w in seed_tokens]
        seed_id = seed_id[::-1]
        if visualiseAttention:
            sentence, attention = predict(seed_id, net_rnn, network_decode, net, encode_seqs2, decode_seqs2, needLogits = False, needAttention = True)
        else:
            sentence = predict(seed_id, net_rnn, network_decode, net, encode_seqs2, decode_seqs2, needLogits = False, needAttention = False)
        print(''.join(sentence))
        if visualiseAttention:
            plt.imshow(np.flip(attention, axis = 1), cmap='gray') # Flip attention as we give a reverse sentence as an input.
            plt.yticks(range(len(sentence)), sentence)
            plt.xticks(range(len(seed_tokens)), seed_tokens, rotation='vertical')
            plt.show()


# In[20]:


# Global parameters
emb_dim = 200
batch_size = 16
n_epoch = 50
T_vocab_size = 50000
generalTokens = ['_', 'unk', '(', ')', ',', 'p', 'a', 'bp', 'path', 'act', 'pmod(P)', 'tloc', 'complex', 'deg', ' -> ', ' -| ', ' -- ', 'PH:placeholder']


# In[22]:


# Download text sentences
TextSentenceID, vocabulary = loadAllTextSentences()

# Download pre-trained word vectors
word_vectors = KeyedVectors.load_word2vec_format('word_embeddings/PubMed-and-PMC-w2v.bin', binary=True)
print('Finish loading word vectors from file') # e.g., word_vectors['increases']

# Create text dict (requiring 1. vocabulary 2. word_vectors 3. T_vocab_size)
T_idx2w, T_w2idx, T_vocab_size_total = createTextDict()

# Create BEL Token dict (requiring 1. generalTokens 2. dicts of each namespace)
B_idx2w, B_w2idx, B_vocab_size_total = createBELTokenDict()

# Download (or creat, if not exists) word_embedding
word_embedding = loadWordEmbeddings()

# Download (or creat, if not exists) BELToken_embedding
BELToken_embedding = loadBELTokenEmbeddings()

# Load training and validation data
trainT, trainB, trainZipsort = loadTrainingData('dataset/TrainingNormalised.BEL')
sampleT, sampleB, sampleZipSort = loadTrainingData('dataset/SampleSetNormalised.BEL')


# In[23]:


# Define models (for train, test, validate)
g = tf.Graph()
with g.as_default() as graph:
    tl.layers.clear_layers_name()

    # model for training
    encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs") # encoding input  ['It', 'was', 'choking', 'with', 'smoke', '.', '_', '_']
    decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs") # decoding input  ['start_id', 'Nó', 'đặc', 'khói', '.', '_']
    target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs") # decoding output ['Nó', 'đặc', 'khói', '.', 'end_id', '_']
    target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask") # tl.prepro.sequences_get_mask()
    net_out, _, net_encode, _, _, net_decode = modelWithAttention(encode_seqs, decode_seqs, is_train=True, reuse=False)

    # model for inferencing
    encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
    decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")
    net, net_rnn, _, network_decode, cell, _ = modelWithAttention(encode_seqs2, decode_seqs2, is_train=False, reuse=True)
    y = tf.nn.softmax(net.outputs)
    
    # model for validation
    encode_seqs3 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
    decode_seqs3 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")
    target_seqs3 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="target_seqs")
    target_mask3 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="target_mask")
    net_val, val_network_encode, _, val_network_decode, _, _  = modelWithAttention(encode_seqs3, decode_seqs3, is_train=False, reuse=True)


# In[24]:


# Define loss function and optimisation
with g.as_default() as graph:
    loss = tl.cost.cross_entropy_seq_with_mask(logits=net_out.outputs, target_seqs=target_seqs, input_mask=target_mask, return_details=False, name='cost')
    lr = 0.001
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    print_all_variables(train_only=True)


# In[33]:


# Initialise models
with g.as_default() as graph:
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.assign_params(sess, [word_embedding], net_encode)
    tl.files.assign_params(sess, [BELToken_embedding], net_decode)
#     tl.files.load_and_assign_npz(sess=sess, name='T2BWithLuongAttention.npz', network=net)
    tl.files.load_and_assign_npz(sess=sess, name='T2BWithLuongAttentionDecode.npz', network=net)
    tl.files.load_and_assign_npz(sess=sess, name='T2BWithLuongAttentionEncode.npz', network=net_rnn)
    b = get_variables_with_name(name='model/memory_layer/kernel:0')
    c = tl.files.load_npz(path='', name='memory_layer.npz')
    op_assign = b[0].assign(c[0])
    sess.run(op_assign)


# In[34]:


# Train
with g.as_default() as graph:
    start_id = B_vocab_size_total-2
    end_id = B_vocab_size_total-1
    n_step = int(len(trainT)/batch_size)
    for epoch in range(n_epoch):
        epoch_time = time.time()
        ## shuffle training data
        trainT, trainB = shuffle(trainT, trainB, random_state=0)
        ## train an epoch
        total_err, n_iter = 0, 0
        for X, Y in tl.iterate.minibatches(inputs=trainT, targets=trainB, batch_size=batch_size, shuffle=False):
            step_time = time.time()

            X = tl.prepro.pad_sequences(X)
            X = [x[::-1] for x in X] # Reverse sequence
            _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
            _target_seqs = tl.prepro.pad_sequences(_target_seqs)

            _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
            _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
            _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

            ## you can view the data here
    #         for i in range(len(X)):
    #             print(i, [E_idx2w[id] for id in X[i]])
    #             print(i, [V_idx2w[id] for id in _target_seqs[i]])
    #             print(i, [V_idx2w[id] for id in _decode_seqs[i]])
    #             print(i, _target_mask[i])
    #             print(len(_target_seqs[i]), len(_decode_seqs[i]), len(_target_mask[i]))
            # exit()

            _, err = sess.run([train_op, loss],
                            {encode_seqs: X,
                            decode_seqs: _decode_seqs,
                            target_seqs: _target_seqs,
                            target_mask: _target_mask})

            if n_iter % 10 == 0:
                print("Epoch[%d/%d] step:[%d/%d] loss:%f took:%.5fs" % (epoch, n_epoch, n_iter, n_step, err, time.time() - step_time))

            total_err += err; n_iter += 1

            ###============= inference
            if n_iter % 50 == 0:
                testInference()

        print("Epoch[%d/%d] averaged loss:%f took:%.5fs" % (epoch, n_epoch, total_err/n_iter, time.time()-epoch_time))
        
        tl.files.save_npz(net.all_params, name='T2BWithLuongAttentionDecodePretrainedTokenEmb.npz', sess=sess)
        tl.files.save_npz(net_rnn.all_params, name='T2BWithLuongAttentionEncodePretrainedTokenEmb.npz', sess=sess)
        b = get_variables_with_name(name='model/memory_layer/kernel:0')
        tl.files.save_npz(b, name='memory_layer_PretrainedTokenEmb.npz', sess=sess)
        if epoch % 2 == 0:
            print("Validation Results - Epoch: %d, Error: %f" % (epoch, validate()))


# In[ ]:


testInference(visualiseAttention = True)


# In[ ]:


# sentences = loadSentences('dataset/TrainingNormalised.BEL')
# trainBSentences = [line['BEL-normalised'] for line in sentences]
# trainBTokenized = [tokeniseBEL(line) for line in trainBSentences]
# trainTSentences = [TextSentenceID[line['Sentence-ID'][4:]]['text'] for line in sentences]
# trainTTokenized = [TextSentenceID[line['Sentence-ID'][4:]]['tokens'] for line in sentences]
# assert len(trainBTokenized) == len(trainTTokenized)
# print(trainBTokenized[150], trainTTokenized[150])


# In[23]:


# Use function loadTrainingData instead
# sentences = loadSentences('dataset/SampleSetNormalised.BEL')
# sampleBSentences = []
# sampleBTokenized = []
# sampleTSentences = []
# sampleTTokenized = []
# for line in sentences:
#     BELTokens = tokeniseBEL(line['BEL-normalised'], mapToHGNC = True)
#     if not BELTokens:
#         print('Deleted BEL sentence:', line['BEL-normalised'])
#         continue
#     sampleBSentences.append(line['BEL-normalised'])
#     sampleBTokenized.append(BELTokens)
#     sampleTSentences.append(TextSentenceID[line['Sentence-ID'][4:]]['text'])
#     sampleTTokenized.append(TextSentenceID[line['Sentence-ID'][4:]]['tokens'])
# sampleT1 = [words2index(sublist, T_w2idx) for sublist in sampleTTokenized]
# sampleB1 = [words2index(sublist, B_w2idx) for sublist in sampleBTokenized]
# assert len(sampleT1) == len(sampleB1)
# zipsort = sorted(zip(sampleT1,sampleB1), key=lambda pair: len(pair[0]))
# sampleT = [x for x, y in zipsort]
# sampleB = [y for x, y in zipsort] # Sort to make a batch have sentences with similar lengths
# sampleZipSort = zipsort
# print(sampleB[10], sampleT[10])


# In[ ]:


# # model for training
# encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs") # encoding input  ['It', 'was', 'choking', 'with', 'smoke', '.', '_', '_']
# decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs") # decoding input  ['start_id', 'Nó', 'đặc', 'khói', '.', '_']
# target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs") # decoding output ['Nó', 'đặc', 'khói', '.', 'end_id', '_']
# target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask") # tl.prepro.sequences_get_mask()
# net_out, _, net_encode = model(encode_seqs, decode_seqs, is_train=True, reuse=False)

# # model for inferencing
# encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
# decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")
# net, net_rnn, _ = model(encode_seqs2, decode_seqs2, is_train=False, reuse=True)
# y = tf.nn.softmax(net.outputs)


# In[ ]:


# Previous version of decoding for test data (Now, use the testModel function)
# sentences = loadSentences('dataset/Task1NeuV3_corrected.sentence')
# sampleTSentences = [TextSentenceID[line['Sentence-ID'][4:]]['text'] for line in sentences]
# sampleTTokenized = [TextSentenceID[line['Sentence-ID'][4:]]['tokens'] for line in sentences]
# sampleT = [words2index(sublist, T_w2idx) for sublist in sampleTTokenized]
# f = open('results/Task1NeuV3_corrected' + '_' + strftime("%Y%m%d%H%M%S", gmtime()) +'.txt', 'w')
# for i in range(len(sampleT)):
#     sentence_id = sentences[i]['Sentence-ID'][4:]
#     seed_id = sampleT[i]
#     # 1. encode, get state
#     state = sess.run(net_rnn.final_state_encode,
#                     {encode_seqs2: [seed_id]})
#     # 2. decode, feed start_id, get first word
#     #   ref https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py
#     o, state = sess.run([y, net_rnn.final_state_decode],
#                     {net_rnn.initial_state_decode: state,
#                     decode_seqs2: [[start_id]]})
#     w_id = tl.nlp.sample_top(o[0], top_k=1)
#     w = B_idx2w[w_id]
#     # 3. decode, feed state iteratively
#     sentence = [w]
#     for _ in range(30): # max sentence length
#         o, state = sess.run([y, net_rnn.final_state_decode],
#                         {net_rnn.initial_state_decode: state,
#                         decode_seqs2: [[w_id]]})
#         w_id = tl.nlp.sample_top(o[0], top_k=1)
#         w = B_idx2w[w_id]
#         if w_id == end_id:
#             break
#         sentence = sentence + [w]
#     f.write(sentence_id + '\t' + ''.join(sentence) + '\n')
# f.close()


# In[ ]:


# Old Inference code
# seeds = ["MMP-2 gelatinolytic activity was higher in cells infected with PBS (mock) and Ad-SV.", 
#                          # p(HGNC:MMP2) increases act(p(HGNC:MMP2))
#                         "In contrast, vandetanib treatment induced a 2.3-fold increase in eNOS mRNA in B16.F10 vasculature.", 
#                          # kin(p(MGI:Kdr)) decreases p(MGI:Nos3)
#                         "Thus, extramitochondrially targeted AIF is a dominant cell death inducer.", 
#                          # p(MGI:Aifm1) increases bp(GOBP:"cell death")
#                         "Binding of PIAS1 to human AR DNA+ligand binding domains was androgen dependent in the yeast liquid beta-galactosidase assay.", 
#                          # a(CHEBI:androgen) -> complex(p(HGNC:AR),p(HGNC:PIAS1))"
#                         "The data suggest that genistein may inhibit CFTR by two mechanisms.", 
#                          # a(CHEBI:genistein) decreases p(MGI:Cftr)
#                         "LPS-induced NO synthesis feedback regulates itself through up-regulation of OPN promoter activity and gene transcription." 
#                          # a(CHEBI:lipopolysaccharide) increases a(CHEBI:"nitric oxide"), p(MGI:Spp1) decreases a(CHEBI:"nitric oxide") 
#                         ] 
#                 for seed in seeds:
#                     print("Input >", seed)
#                     output = nlp.annotate(seed, properties={'annotators': 'tokenize', 'outputFormat': 'json'})
#                     seed_tokens = [i['originalText'] for i in output['tokens']]
#                     seed_id = [T_w2idx[w] if w in T_w2idx else T_w2idx['unk'] for w in seed_tokens]
#                     seed_id = seed_id[::-1]

#     #                 # 1. encode, get state
#     #                 state = sess.run(net_rnn.final_state_encode,
#     #                                 {encode_seqs2: [seed_id]})
#     #                 # 2. decode, feed start_id, get first word
#     #                 #   ref https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py
#     #                 o, state = sess.run([y, net_rnn.final_state_decode],
#     #                                 {net_rnn.initial_state_decode: state,
#     #                                 decode_seqs2: [[start_id]]})
#     #                 w_id = tl.nlp.sample_top(o[0], top_k=1)
#     #                 w = B_idx2w[w_id]
#     #                 # 3. decode, feed state iteratively
#     #                 sentence = [w]
#     #                 for _ in range(30): # max sentence length
#     #                     o, state = sess.run([y, net_rnn.final_state_decode],
#     #                                     {net_rnn.initial_state_decode: state,
#     #                                     decode_seqs2: [[w_id]]})
#     #                     w_id = tl.nlp.sample_top(o[0], top_k=1)
#     #                     w = B_idx2w[w_id]
#     #                     if w_id == end_id:
#     #                         break
#     #                     sentence = sentence + [w]
#     #                 print(" >", ' '.join(sentence))

#                     # 1. encode, get state
#                     state, memory = sess.run([net_rnn.final_state, net_rnn.outputs],
#                                     {encode_seqs2: [seed_id]})
# #                     print(state)
#                     # 2. decode, feed start_id, get first word
#                     #   ref https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py
#                     feed_dict = {network_decode.initial_state.cell_state:state,
#                                  network_decode.memory:memory,
# #                                     encode_seqs2: [seed_id],
#                                     decode_seqs2: [[start_id]]}
# #                     print(feed_dict)
#                     o, state = sess.run([y, network_decode.final_state],
#                                     feed_dict)
# #                     print("state", state)
# #                     print("outputs", outputs)
#                     w_id = tl.nlp.sample_top(o[0], top_k=1)
#                     w = B_idx2w[w_id]
#                     # 3. decode, feed state iteratively
#                     sentence = [w]
#                     for _ in range(30): # max sentence length
#                         o, state = sess.run([y, network_decode.final_state],
#                                         {network_decode.initial_state:state,
#                                          network_decode.memory:memory,
# #                                          encode_seqs2: [seed_id],
#                                         decode_seqs2: [[w_id]]})
#                         w_id = tl.nlp.sample_top(o[0], top_k=1)
#                         w = B_idx2w[w_id] ###
#                         if w_id == end_id:
#                             break
#                         sentence = sentence + [w]
#                     print(" >", ' '.join(sentence))

