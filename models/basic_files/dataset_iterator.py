# Dataset iterator file.
import random
import nltk
import numpy as np
import pickle
import sys
import os.path
import tensorflow as tf
import copy
from .vocab import *
from nltk.corpus import stopwords as sw
import string
import re
import itertools

def helper_function(content_file, summary_file, query_file, model):
    content = open(content_file).readlines()
    summary = open(summary_file).readlines()
    query = open(query_file).readlines()
    closest_pairs = {}
    stopwords = sw
    new_content, new_summary, new_query = [], [], []
    stopwords = set(stopwords.words('english') + ["<eos>", "<s>"] + string.punctuation.split() + ";:><".split())
    data = zip(content, summary, query)

    if os.path.exists("__words.pkl"):
        __words = pickle.load(open("__words.pkl", "rb"))

    for c, s, q in data:
        c = " ".join(c.strip().split()[1:-1])
        s = " ".join(s.strip().split()[1:-1])
        q = " ".join(q.strip().split()[1:-1])
        c =  re.sub("\d","#",c)
        s = re.sub("\d","#",s)
        q = re.sub("\d","#",q)
        temp_c = c.strip().split()
        temp_s = s.strip().split()
        temp_q = q.strip().split()
        new_content.append(" ".join(temp_c))
        new_summary.append(" ".join(temp_s))
        new_query.append(" ".join(temp_q))
        c_without_sw = set(temp_c + temp_s + temp_q)- set(stopwords)
        s_without_sw = set(temp_s) - set(stopwords)
        q_without_sw = set(temp_q) -set(stopwords)
    
        if len(c_without_sw) < 25:
            continue

        for k in range(12,25):
            new_c = []
            new_q = []
            new_s = []

            number_of_words_replace = k
            words_replace_c = random.sample(set(c_without_sw), number_of_words_replace)

            if len(set(s_without_sw)) >= 3:
                words_replace_s = random.sample(set(s_without_sw),random.randint(1,3))
            elif len(set(s_without_sw)) >=1 :
                words_replace_s = random.sample(set(s_without_sw),1)
            else:
                words_replace_s = []

            if len(set(q_without_sw)) >= 3:
                words_replace_q = random.sample(set(q_without_sw), random.randint(1,3))
            elif len(set(s_without_sw)) >=1:
                words_replace_q = random.sample(set(q_without_sw),1)
            else:
                words_replace_q = []

            for w in temp_c:
                if w in words_replace_c:
                    if w not in __words and w in model:
                        __words[w] = model.most_similar(w)[0][0]
                    if w in __words:
                        new_c.append(__words[w])
                    else:
                        new_c.append(w)
                else:
                    new_c.append(w)

            new_s = temp_s

            for w in temp_q:
                if w in words_replace_c + words_replace_q:
                    if w not in __words and w in model:
                        __words[w] = model.most_similar(w)[0][0]
                    if w in __words:
                        new_q.append(__words[w])
                    else:
                        new_q.append(w)
                else:
                    new_q.append(w)

            new_content.append("<s> " + " ".join(new_c) + " <eos>")
            new_summary.append("<s> " + " ".join(new_s) + " <eos>")
            new_query.append("<s> " + " ".join(new_q) + " <eos>")

    x = list(zip(new_content, new_summary, new_query))
    random.shuffle(x)
    cont , summ, quer = [], [], []
    for c, s, q in x:
        cont.append(c)
        summ.append(s)
        quer.append(q)

    if (os.path.exists('__words.pkl')) == False:
        with open("__words.pkl", "wb") as f:
            pickle.dump(__words, f)

    print ("Done")
    return cont, summ, quer

class Datatype:

    def __init__(self, name, title, label, content, query, num_samples, content_sequence_length, \
                 query_sequence_length,\
                 max_length_content, max_length_title, max_length_query, seqindices_encoder,
                 seqindices_query):
        """ Defines the dataset for each category valid/train/test

        Args:
            name   : Name given to this partition. For e.g. train/valid/test
            title  : The summaries that needs to be generated.
            content: The input/source documents 
            query  : The queries given based on which the document needs to be summarized

            num_samples        :  Number of samples in this partition
            max_length_content :  Maximum length of source document across all samples
            max_length_title   :  Maximum length of summary across all samples
            
            global_count_train : pointer to retrieve the next batch during training
            global_count_test  : pointer to retrieve the next batch during testing
        """

        self.name    = name
        self.title   = title
        self.content = content
        self.labels  = label
        self.query   = query

        self.content_sequence_length = content_sequence_length
        self.query_sequence_length   = query_sequence_length

        self.number_of_samples  = num_samples
        self.max_length_content = 122#max_length_content
        self.max_length_title   = 10 #10#max_length_title - 1
        self.max_length_query   = 21#max_length_query

        self.global_count_train = 0
        self.global_count_test  = 0

        self.sequence_indices_encoder = seqindices_encoder
        self.sequence_indices_query = seqindices_query


class PadDataset:

    def pad_data(self, data, max_length):
        """ Pad the batch to max_length given.

            Arguments: 
                data       : Batch that needs to be padded
                max_length : Max_length to which the samples needs to be
                             padded.

            Returns:
                padded_data : Each sample in the batch is padded to 
                              make it of length max_length.
        """

        padded_data = []
        sequence_length_batch = []
        for lines in data:
            if (len(lines) < max_length):
                temp = np.lib.pad(lines, (0,max_length - len(lines)),
                    'constant', constant_values=0)

                sequence_length_batch.append(len(lines))
            else:
                temp = lines[:max_length]
                sequence_length_batch.append(max_length)

            padded_data.append(temp)

        return padded_data, sequence_length_batch


    def make_batch(self, data, batch_size, count, max_length, sequence_data=None):
        """ Make a matrix of size [batch_size * max_length]
            for given dataset

            Args:
                data: Make batch from this dataset
                batch_size : batch size
                count : pointer from where retrieval will be done
                max_length : maximum length to be padded into

            Returns
                batch: A matrix of size [batch_size * max_length]
                count: The point from where the next retrieval is done.
        """

        batch = []
        sequence_indices_batch = []
        batch = data[count:count+batch_size]
        if sequence_data is not None:
            sequence_indices_batch = sequence_data[count:count+batch_size]
        count = count + batch_size


        while (len(batch) < batch_size):
            batch.append(np.zeros(max_length, dtype = int))
            sequence_indices_batch.append(np.zeros(max_length, dtype=int))
            count = 0
            
        batch, sequence_length_batch = self.pad_data(batch,max_length)
        sequence_indices_batch, _ = self.pad_data(sequence_indices_batch, max_length)
        batch = np.transpose(batch)
        sequence_indices_batch = np.transpose(sequence_indices_batch)
        if sequence_data is None:
            sequence_indices_batch = None
        return batch, count, sequence_length_batch, sequence_indices_batch

    def make_batch_sequence(self, data, batch_size, count, max_length):
        """ Make a matrix of size [batch_size * max_length]
            for given dataset

            Args:
                data: Make batch from this dataset
                batch_size : batch size
                count : pointer from where retrieval will be done
                max_length : maximum length to be padded into

            Returns
                batch: A matrix of size [batch_size * max_length]
                count: The point from where the next retrieval is done.
        """

        batch = []
        batch = data[count:count+batch_size]
        count = count + batch_size


        while (len(batch) < batch_size):
            batch.append(np.zeros(max_length, dtype = int))
            count = 0

        return batch, count

    def next_batch(self, dt, batch_size, c=True):
        """ Creates a batch given the batch_size from
            mentioned dataset iterator.

            Arguments:
              * dt: Datatset from which the batch needs to
                    retreived
              * batch_size: Number of samples to keep in a batch

            Returns:
              * batch: Returns the batch created
        """

        if (c is True):
            count = dt.global_count_train
        
        else:
            count = dt.global_count_test

        temp_data = {}

        max_length_content = max(self.datasets[i].max_length_content for i in self.datasets)
        max_length_title   = max(self.datasets[i].max_length_title   for i  in self.datasets)
        max_length_query   = max(self.datasets[i].max_length_query   for i in self.datasets)

        contents, count1, content_sequence_length, sequence_indices_encoder = self.make_batch(dt.content, batch_size, count, max_length_content, dt.sequence_indices_encoder)
        titles,   _ ,_ ,_    = self.make_batch(dt.title,   batch_size, count,   max_length_title)
        labels,   _ ,_ ,_    = self.make_batch(dt.labels,  batch_size, count,   max_length_title)
        query,    _ , query_sequence_length, sequence_indices_query    = self.make_batch(dt.query,  batch_size, count,   max_length_query, dt.sequence_indices_query)

        # Weights for the loss function for the decoder
        weights = copy.deepcopy(titles)


        # Fill the weighs matrix, based on the label parameters.
        for i in range(titles.shape[0]):
            for j in range(titles.shape[1]):
                if (weights[i][j] > 0):
                        weights[i][j] = 1
                else:
                        weights[i][j] = 0

        if (c == True): 
            dt.global_count_train = count1 % dt.number_of_samples
        else:
            dt.global_count_test  = count1 % dt.number_of_samples

        temp_data["encoder_inputs"] = contents
        temp_data["decoder_inputs"] = titles
        temp_data["labels"] = labels
        temp_data["weights"] = weights
        temp_data["query"] = query
        temp_data["query_seq_length"] = query_sequence_length
        temp_data["encode_seq_length"] = content_sequence_length
        temp_data["sequence_indices_encoder"] = sequence_indices_encoder
        temp_data["sequence_indices_query"] = sequence_indices_query
        return temp_data

    def load_data_helper(self, wd="../Data/"):
        t_cont = os.path.join(wd, "train_content")
        t_title = os.path.join(wd, "train_summary")
        t_query = os.path.join(wd, "train_query")
        content, title, query = helper_function(t_cont, t_title, t_query, self.vocab.embeddings_model)
        return content, title, query

    def load_data_file(self,name, title_file, content_file, query_file, count):
        """ Each of the (train/test/valid) is loaded separately.

        Arguments:
        * title_file   : The file containing the summaries
                * content_file : The file containing the source documents
                * query_file   : The file containing the queries


           Returns:
           * A Datatype object that contains relevant information to 
                 create batches from the given dataset
 
        """
        if (name == "train"):
            content, title, query = content_file, title_file, query_file

        else:
            content = open(content_file).readlines()
            title   = open(title_file).readlines()
            query   = open(query_file).readlines()
        title_encoded   = []
        content_encoded = []
        label_encoded   = []
        query_encoded   = []
        seqindices_encoder = []
        seqindices_query = []

        content_sequence_length  = []
        query_sequence_length    = []
        
        max_title = 0
        for lines in title:
            temp = [self.vocab.encode_word_decoder(word) for word in lines.split()]

            if (len(temp) > max_title):
                max_title = len(temp)

            title_encoded.append(temp[:-1])
            label_encoded.append(temp[1:])

        max_content = 0

        for lines, ind in zip(content, range(count, len(content) + count)):
            temp = [self.vocab.encode_word_encoder(word) for word in lines.split()]

            if (len(temp) > max_content):
                max_content = len(temp)

            content_encoded.append(temp)
            seqindices_encoder.append([ind]*len(temp))
            content_sequence_length.append(len(temp))

        max_query = 0
        for lines, ind in zip(query, range(count, len(query) + count)):
            temp = [self.vocab.encode_word_encoder(word) for word in lines.split()]

            if (len(temp) > max_query):
                max_query = len(temp)

            query_encoded.append(temp)
            seqindices_query.append([ind]*len(temp))
            query_sequence_length.append(len(temp))

        return Datatype(name, title_encoded, label_encoded, content_encoded,
                        query_encoded, len(title_encoded), content_sequence_length, query_sequence_length,
                        max_content, max_title, max_query, seqindices_encoder, seqindices_encoder), count + len(content)


    def load_data(self, wd, content, title, query):
        """ Load all the datasets

            Arguments:
        * wd: Directory where all the data files are stored

            Returns:
            * void
        """
        s = wd
        self.datasets = {}
        count = 0
        for i in ("train", "valid", "test"):
            temp_t = s + i + "_summary"
            temp_v = s + i + "_content"
            temp_q = s + i + "_query"
            temp_sie = s + i + "_si_encoder"
            temp_siq = s + i + "_si_query"
            if i == "train":
               self.datasets[i], count = self.load_data_file(i, title, content, query, count)
            else:
              self.datasets[i], count = self.load_data_file(i, temp_t, temp_v, temp_q, count)


    def __init__(self,  working_dir = "../Data/", embedding_size=100, global_count = 0, diff_vocab = False,\
                 embedding_path="../Data/embeddings.bin", limit_encode = 0, limit_decode=0,
                 embedding_sequence_encoder_path=None, embedding_sequence_query_path=None):
        """ Create the vocabulary and load all the datasets

            Arguments:
        * working_dir   : Directory path where all the data files are stored
        * embedding_size: Dimension of vector representation for each word
        * diff_vocab    : Different vocab for encoder and decoder. 

        Returns:
        * void

        """

        filenames_encode = [ working_dir + "train_content", working_dir + "train_query" ]
        filenames_decode = [ working_dir + "train_summary" ]

        self.global_count = 0
        self.vocab        = Vocab()

        if (diff_vocab == False):
            filenames_encode = filenames_encode + filenames_decode
            filenames_decode = filenames_encode

        self.vocab.get_global_embeddings(embedding_size, embedding_path)
        content, title, query = self.load_data_helper(working_dir)

        lines_embed_content = content + open(os.path.join(working_dir , "valid_content")).readlines()  + open(os.path.join(working_dir, "test_content")).readlines()
        lines_embed_query = query + open(os.path.join(working_dir , "valid_query")).readlines()  + open(os.path.join(working_dir, "test_query")).readlines()

        self.vocab.construct_vocab(content + query + title , title,  embedding_size, embedding_path,
                                   limit_encode, limit_decode, lines_embed_content, lines_embed_query)

        self.load_data(working_dir, content, title, query)
        #print (self.vocab.word_to_index_decode)



    def length_vocab_encode(self):
        """ Returns the encoder vocabulary size
        """
        return self.vocab.len_vocab_encode

    def length_vocab_decode(self):
        """ Returns the decoder vocabulary size
        """
        return self.vocab.len_vocab_decode

    def decode_to_sentence(self, decoder_states):
        """ Decodes the decoder_states to sentence
        """
        s = ""
        for temp in (decoder_states):
            word = self.vocab.decode_word_decoder(temp)
            s = s + " " + word

        return s

