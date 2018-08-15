#Remove the data points, where the query is very generic.

import os
import sys
import shutil


def get_content(filename, f1):

	with open(filename, "r") as f:
		i = f.read()
		f1.write(i + "\n")


f1 = open("temp_query_content","w")

x = sys.argv[1]

with open(x, "r") as f:
	for lines in f:
		get_content(lines.strip(), f1)


f1 = open(sys.argv[1], "r")
f2 = open("temp_query_content", "r")

for (l1, l2) in zip(f1,f2):
	print l1
	l2 = l2.lower()

	if (l2.isspace()) or ('pro/con' in l2) or ('subquestion here' in l2) or ('sub question here' in l2) or ('Videos' in l2) or ('Pro and con videos' in l2) or ('argument #' in l2) or (l2.startswith("<q>argument")):
		shutil.rmtree(l1.strip()[:-6])


os.remove("temp_query_content")
