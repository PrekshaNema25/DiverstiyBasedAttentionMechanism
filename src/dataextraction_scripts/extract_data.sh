# Run the bash script to extract the data from debatepedia website.

# Give the links to the categories in the "categories" file, the debate links will be extracted 
# These links will be stored in category_links/all_links
python extract_links.py debatepedia_categories
#python extract_links.py c
echo "Extracted links"

#To extract the document/summary/query from the output of the above command:
#The text files containing the above mentioned information will be stored in the 
#director Data
python extract_text.py category_links/all_links Data
echo "Extracted the text from the links"

# Store the path of al the query files.
find Data/ -name query > query_paths
echo "Query paths retreived"

#query_paths is used to prune out the irrelevant queries.
python prune_empty_queries.py query_paths
echo "Pruned unnecessary queries"

find Data/ -type d -empty -delete
echo "Deleted some of the empty directories"

python combine_data.py Data new_Data
echo "Combine the debates to one file"

python preprocessed_data.py new_Data/content.txt new_Data/summary.txt new_Data/query.txt
echo "Preprocessed the data"

mv final_* new_Data

python make_folds.py new_Data/final_content new_Data/final_summary new_Data/final_query 10 10_folds
echo "Make 10 folds"

