# Text-based Lookalike modeling of users web traffic to build a recommender
A Python based project that experiments on different machine learning techniques to model user generated textual data (e.g. visited urls, comments, etc) to build a recommender system in applications like advertizement, tailored suggestion, etc. When the test case is a user's text data, the outcome is finding the similar user and illustraing the behaviour in a regressive analytical basis. 

# Prerequisites
- Python3.7
- Packages: numpy, pandas, matplotlib, sklearn, nltk, keras, tensorflow
- Appropriately formated data (no data is given here due to copyright issue)

# Algorithm description
- Read and understand the data (exploratory data analysis)
- Clean the data
- Group by the data based on each user_id by concatenating page_urlpath to create feature vector 
- Again by taking probability of click data from seed column by averaging for each user_id to create lookup table for test case probability at the end
- Feature extraction from concatenated page_urlpath using guide:  
