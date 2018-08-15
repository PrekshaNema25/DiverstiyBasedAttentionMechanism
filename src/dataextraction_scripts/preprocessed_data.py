import os
import sys
from itertools import *
import random
import string
import nltk
import re
import numpy as np
from nltk.tokenize import WhitespaceTokenizer 

def remove_punctuation(s):
        punc = string.punctuation

        replace_punctuation = string.maketrans(punc, ' '*len(punc))
        s = str(s).translate(replace_punctuation)
        return s

def preprocess(s, max_tokens):
    #s = unicode(s, ignore="errors")
    s = s.lower()
    s = re.sub(r'[^\x00-\x7F]+',' ', s)
    s = re.sub("<s>", "", s)
    s = re.sub("<eos>", "", s)
    s = remove_punctuation(s)
    s = re.sub('\d','#',s)
    s = re.sub('\n',' ',s)
    s = re.sub(',',' ',s)
    
    tokens = WhitespaceTokenizer().tokenize(s)
    #s = replace_the_unfrequent(tokens)
    if (len(tokens) > max_tokens):
	tokens = tokens[:max_tokens]

    s = " ".join(tokens)
    return s, len(tokens)

def preprocess_lines(filename, max_tokens):

	m = filename + "_m"
	f1 = open(m, "w")

	with open(filename, "r") as f:
		m1 = []
		for lines in f:
			s, m = preprocess(lines, max_tokens)
			m1.append(m)
			f1.write(s + "\n")

	print ("Average words", np.max(m1))
	f1.close()


def shuffle_data(content, summary, query, name):

	f1 = open(content, "r")
	f2 = open(summary, "r")
	f3 = open(query, "r")

	l1 = f1.read().splitlines()
	l2 = f2.read().splitlines()
	l3 = f3.read().splitlines()

	print (len(l1))
	print(len(l2))
	print (len(l3))

	combine = zip(l1,l2,l3)
	#print combine	
	random.shuffle(combine)
	#print combine

	print name
	with open(name + "_content", "w") as w1, open(name+"_summary", "w") as w2, open(name+"_query", "w") as w3:

		for c in combine:
			#print c
			if (len(c[0].split()) < 4):
				continue

			if not(c[0].strip() == "" or c[1].strip() == ""  or c[2].strip() == ""):
				w1.write(  "<s> "+  c[0] +" <eos>\n")
				w2.write(  "<s> " + c[1] +" <eos>\n")
				w3.write(  "<s> " + c[2] +" <eos>\n")

def main():
	content = sys.argv[1]
	summary = sys.argv[2]
	query   = sys.argv[3]
	preprocess_lines(content, 120)
	preprocess_lines(summary, 20)
	preprocess_lines(query, 20)

	c1 = sys.argv[1] + "_m"
	c2 = sys.argv[2] + "_m"
	c3 = sys.argv[3] + "_m"

 	shuffle_data(c1, c2, c3, "final")

	os.remove(c1)
	os.remove(c2)
	os.remove(c3)

	
if __name__ == '__main__':
 	main()

		
