# Author: Preksha Nema
#Divide the data into number of folds specified.
# Argument #1 : Source File (Content file)
# Arguemnt #2 : Summary File (True label file)
# Argument #3 : Query File
#Argument #4: Number of folds in which the mentioned files will be divided.
# Argument #5: Directory where the 10 folds will be stored.
import sys 
import os


def fill_the_lists(filename):

	f = open(filename, "r")

	l = []
	for line in f:
		l.append(line)

	return l


def fill(content, summary, query):

	c = fill_the_lists(content)
	s = fill_the_lists(summary)
	q = fill_the_lists(query)

	return c, s, q


def make_different_datasets(content, summary, query, number_of_folds, name):

	i = 0
	j = 0
	c, s, q = fill(content, summary, query)

	exms_valid = len(c)/number_of_folds
	exms_test = len(c)/number_of_folds
	exms_train = len(c) - exms_test - exms_valid

	for count in range(1,number_of_folds+1):

		if not (os.path.exists(os.path.join(name, str(count)))):
			os.makedirs(os.path.join(name, str(count)))

		i = j
		print i
		t = 0
		valid_content = []
		valid_summary = []
		valid_query = []
		while (t < exms_valid):

			if (i >= len(c)):
				i = 0

			valid_content.append(c[i])
			valid_query.append(q[i])
			valid_summary.append(s[i])
			i = i + 1
			t = t + 1

		t = 0
		test_content = []
		test_summary = []
		test_query = []
		while(t < exms_test):

			if (i >= len(c)):
				i = 0

			test_content.append(c[i])
			test_query.append(q[i])
			test_summary.append(s[i])
			i = i + 1
			t = t + 1

		t = 0
		train_content = []
		train_summary = []
		train_query   = []

		while(t < exms_train):

			if ( i >= len(c)):
				i = 0

			train_content.append(c[i])
			train_query.append(q[i])
			train_summary.append(s[i])
			i = i + 1
			t = t + 1


		vc = open (os.path.join(name, str(count), "valid_content") , "w")
		vs = open ( os.path.join(name, str(count), "valid_summary") , "w")
		vq = open ( os.path.join(name, str(count), "valid_query"), "w")
		trc = open ( os.path.join(name, str(count), "train_content"), "w")
		trs = open ( os.path.join(name, str(count), "train_summary"), "w")
		trq= open ( os.path.join(name, str(count), "train_query"), "w")
		tc = open ( os.path.join(name, str(count), "test_content"), "w")
		ts = open ( os.path.join(name, str(count), "test_summary"), "w")
		tq = open ( os.path.join(name, str(count), "test_query"), "w")

		for m in range(exms_valid):
			vc.write(valid_content[m])
			vs.write(valid_summary[m])
			vq.write(valid_query[m])

		for m in range(exms_test):
			tc.write(test_content[m])
			ts.write(test_summary[m])
			tq.write(test_query[m])

		for m in range(exms_train):
			trc.write(train_content[m])
			trs.write(train_summary[m])
			trq.write(train_query[m])

		j = j + exms_valid

		vc.close()
		vs.close()
		vq.close()
		tc.close()
		ts.close()
		tq.close()
		trc.close()
		trq.close()
		trs.close()


def main():
        data_dir = sys.argv[1]
        content_file = os.path.join(data_dir, "content.txt")
        summary_file = os.path.join(data_dir, "summary.txt")
        query_file   = os.path.join(data_dir, "query.txt")
        make_different_datasets(content_file, summary_file, query_file, int(sys.argv[2]), sys.argv[3])


if __name__ == '__main__':
		main()	



