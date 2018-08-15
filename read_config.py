import sys

class Config:
    """ Config class represents the hyperparameters in a single
        object
    """ 
    def __init__(self, filename="config.txt"):

        """ Initialize the object with the parameters.

        Args:
            learning_rate : Learning rate for the optimizer
            embedding_size: dimensions of word embeddings
            hidden_size   : dimensions of hidden state of rnn cell
            batch_size    : batch size
            max_epochs    : Number of epochs to be run
            early_stop    : early stop

            max_sequence_length_content: Max length to be set for encoder inputs
            max_sequence_length_title  : Max length to be set for decoder inputs
            max_sequence_length_query  : Max length to be set for query inputs
        """

        f = open(filename, "r")
        config_dir = {}
        for line in f:
            if "#" in line or len(line.strip()) == 0:
                continue

            key, value = line.strip().split("=")
            key, value = key.strip(), value.strip()
            if key == "learning_rate":
                config_dir[key] = float(value)
            elif value.isdigit():
                config_dir[key] = int(value)
            elif value == "True":
                config_dir[key] = True
            elif value == "False":
                config_dir[key] = False
            else:
                config_dir[key] = value


        self.config_dir = config_dir
        self.learning_rate  = self.config_dir["learning_rate"]
        self.embedding_size = self.config_dir["embedding_size"]
        self.max_sequence_length_content = self.config_dir["max_sequence_length_content"]
        self.max_sequence_length_title   = self.config_dir["max_sequence_length_title"]
        self.max_sequence_length_query   = self.config_dir["max_sequence_length_query"]
        self.hidden_size = self.config_dir["hidden_size"]
        self.batch_size = self.config_dir["batch_size"]
        self.max_epochs = self.config_dir["max_epochs"]
        self.outdir     = self.config_dir["outdir"]
        self.emb_tr     = self.config_dir["embedding_trainable"]
        self.early_stop = self.config_dir["early_stop"]


def main():
    c = Config("config.txt")
    for k in c.config_dir:
        print (k, c.config_dir[k])

if __name__ == '__main__':
    main()

