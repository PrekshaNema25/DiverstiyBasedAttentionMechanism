# Query Based Abstractive Summarization
## Overview 
Implementation of diversity based attention model for query based abstractive summarization task. The implementation is adapted from the seq2seq package in Tensorflow.

The implementation is based on [this](https://arxiv.org/abs/1704.08300) work.
> Diversity based attention models for Query based abstractive summarization

> Preksha Nema, Mitesh M. Khapra, Anirban Laha, Balaraman Ravindran

> ACL, 2017

## Requirements:
* [tensorflow-1.8](https://www.tensorflow.org/versions/r1.8/)
* [gensim](https://pypi.python.org/pypi/gensim)
* [nltk](http://www.nltk.org/install.html)
* [tensorflow_hub] (https://www.tensorflow.org/hub/modules/google/universal-sentence-encoder/1)

## Data Download and Preprocessing
* One data set split is already provided in data/ folder. 

### To crawl the data from scratch :
* cd src/dataextraction_scripts
* The model will extract the data for the categories mentioned in file debatepedia_categories
* sh extract_data.sh
* python make_folds.py ../../data <number_of_folds> <new_dir_for_10_folds> 
* By default run : python make_folds.py ../../data 10 ../../data
    


## Get the Glove embeddings:
    mkdir Embedding
    cd Embedding
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip glove.840B.300d.zip
    echo 2196017 300 > temp_metadata
    cat temp_metadata glove.840B.300d.txt > embeddings
    rm temp_metadata
    
 ## Configuration file:
     * The hyper parameters could be changed in the config.txt file.
     * The influence of eah hyperarameter have been explained in detail in the comments in config.txt
 ## Train the model
      sh ./train.sh config.txt
      
 ## Inference:
     test.sh will refer to the config.txt to initialize the graph that will be used for creating the graph for inference.
     sh ./test.sh config.txt output/test_final_results
 
 ### If output/test_final_results, exists, just run the following command:
    python postscripts/retreive.py output/test_final_results

## Universal Sentence Encoder:
Including the USE embeddings for document and query sentences, helps in improving the model performance. 
The USE embeddings are concatenated with word embedding and passed through the RNN. Also the sentence embedding is one 
of the vectors based on which the contextuals representations are attended to along with query and decoder state.


