import sys
import string
# Argument 1: The folder that contains all the debate topics
# Argument 2: The folder where the final data will be stored

import os
import re
import sys
import pickle
from itertools import *

def make_files(data_dir, sub_dir, q_file, sum_file, cont_file):

	files = ["content", "summary", "query"]

	for d1 in os.listdir(os.path.join(data_dir, sub_dir)):
		c = os.path.join( data_dir, sub_dir, d1, files[0])
		s = os.path.join( data_dir, sub_dir, d1, files[1])
		q = os.path.join( data_dir, sub_dir, d1, files[2])

		f_q = open(q, "r")
		query_sent = f_q.read().splitlines()[0]

		f_c = open(c, "r")
		f_s = open(s, "r")

		for l1, l2 in izip(f_c, f_s):
			if not(l1.isspace() or l2.isspace()):
				q_file.write(query_sent + "\n")
				sum_file.write(l2 )
				cont_file.write(l1)


def make_files_all_dicts(data_dir, dir_lists, name):

	q_file = open((os.path.join(name ,  "query.txt")), "w")
	c_file = open((os.path.join(name , "content.txt")), "w")
	s_file = open((os.path.join(name , "summary.txt")), "w")

	for d in dir_lists:
		make_files(data_dir, d, q_file, s_file, c_file)


	q_file.close()
	c_file.close()
	s_file.close()

def main():

	data_dir = sys.argv[1]

	dir_list = os.listdir(data_dir)
	print(dir_list)
	name = sys.argv[2]
	
	if not(os.path.isdir(name)):
		os.makedirs(name)


	make_files_all_dicts(data_dir, dir_list, name)

if __name__ == '__main__':
	main()
