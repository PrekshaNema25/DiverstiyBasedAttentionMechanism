from bs4 import *
import bs4
import sys
import urllib2
import os
import requests
from requests.exceptions import HTTPError
from multiprocessing import Pool


def extract_query_links(filename, dirname):

	d = dirname

	all_links = open(filename, "rb")


	e = open('error','wb')
	i = 1
	count = i
	prev_i=1
	for lines in all_links:

		prev_i = i
		if not (os.path.exists(d + "/" + str(count))):
			os.makedirs(d + "/" + str(count))
		dataset = d + "/" + str(count)
		i = extract_single_link(lines, i, dataset)
		if (prev_i == i ):
			e.write(lines)

		count = count  + 1


def get_content(temp):

	context_list = []
	summary_list = []
	while (True):

		if (temp is None or temp.next_sibling is None or temp.next_sibling.next_sibling is None):
			break

		temp = temp.next_sibling.next_sibling

		#print temp
		if (temp is not None):
			if (temp.name == 'ul') or (temp.name == 'p'):
				summary_obj = temp.find('b')

				if (summary_obj is None):
					continue

				context_obj = summary_obj.parent

				context = []

				for i, element in enumerate(context_obj):
					if (i == 0):
						continue
					if isinstance(element, bs4.element.NavigableString):
						#print element

						if element is "\n" or element is " " or element is "":
							continue

						context.append(element)
					else:
						context.append(element.text)

				m = " ".join(w for w in context)
				if (m is None or m.isspace()):
					#print " Preksha"
					c1 = []
					context_obj = context_obj.parent
					#print context_obj
					if context_obj is not None:
						context_obj = context_obj.next_sibling

					while(True):
						if(context_obj is None or context_obj.next_sibling is None):
							break

						temp = context_obj.string
						context_obj = context_obj.next_sibling
						c1.append(temp)

					context = " ".join(w for w in c1 if w is not None)

				else:
					context = m

				summary = ''.join(summary_obj.findAll(text=True))

				summary_list.append(summary)
				context_list.append(context)
	print (context_list, summary_list)
	return context_list, summary_list

def extract_single_link(link, count, dataset):

	try:
		page = urllib2.urlopen(link)

	except:
		print ("Page not downloaded")
		return count

	soup = BeautifulSoup(page.read())

	h3_tags = soup.findAll('h3')

	init = False
	for i in h3_tags[1:]:

		if i.parent is not None and i.parent.parent is not None:
				if i.parent.parent.next_sibling is not None:
						if (i.parent.parent.next_sibling.next_sibling is not None):

							h4_tags = i.parent.parent.next_sibling.next_sibling.findAll('h4')

							if (type(i) is not bs4.element.Tag):
								continue

							if not all(c.name is None for c in i.children):
								query = ''.join(i.findAll(text=True))

							else:
								query = i.string


							init = False
							output_path = dataset + "/" + str(count)
							
							if os.path.exists(output_path) == False:
								os.mkdir(output_path)

							for j in h4_tags:

								if (('Yes' in j.string or 'Pro' in j.string) or ('No' in j.string or 'Con' in j.string)):

									temp = j

									context_list, summary_list = get_content(temp)

									if (init==False):
										print (output_path)
										output_file = open(output_path + "/query" , "wb")
										output_summary = open(output_path +"/summary","wb")
										output_content = open(output_path +"/content","wb")
										output_file.write(query.encode('utf-8') )
										init = True
										count = count + 1

									for context in context_list:
										context = unicode(context)
										output_content.write(  context.encode('utf-8').strip('\n') + "\n")

									for summary in summary_list:
										summary= unicode(summary)
										output_summary.write( summary.encode('utf-8') +"\n")
									
	return count


def main():
	extract_query_links(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
		main()	
