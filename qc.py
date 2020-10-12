import os
import re
import sys
import sklearn as skl
import numpy as np
import nltk

#-----------------------------------------------------------------
# Global Variable Field 
#-----------------------------------------------------------------

# Stop words list from nltk for the English language
stop_words_nltk_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
"you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 
'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 
'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 
'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 
"aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

# TODO: Choose which stopwords we want to actually remove
stop_words = []
training_data = {}
index_training_data = {} 

#-----------------------------------------------------------------
# Stemming and/or Lemmatization of a line 
#-----------------------------------------------------------------
def stem_lem(line):
    # Uncomment if necessary
    #nltk.download('wordnet')

    # TODO: Check type of Stemmer 
    # stemmer = nltk.stem.snowball.EnglishStemmer()
    # p_line = [stemmer.stem(word) for word in line]

    # Lemmer  
    lemmer = nltk.WordNetLemmatizer()
    p_line = [lemmer.lemmatize(word) for word in line]

    return p_line

#-----------------------------------------------------------------
# Preprocesses a line 
#-----------------------------------------------------------------
def preprocess_line(line):
    # Lowercasing the entire line
    p_line = line.lower()
    
    # Remove stop words (we Tokenize implicity to make it easier)
    for word in nltk.word_tokenize(p_line):
        if word in stop_words:
            p_line = re.sub('{}\s'.format(word),'', p_line)

    # Remove punctuation using regex
    p_line = re.sub('[?!\.,;:`\']','',  p_line)

    # TODO: Check this remove the end of 's. The ownership relation it conveys
    # might be worthwhile to keep. 
    p_line = re.sub('\ss\s',' ',  p_line)

    # Tokenization 
    p_line = nltk.word_tokenize(p_line)   

    # TODO: ? Fix spelling mistakes

    # TODO: Check this
    # Stemmerization and Lemmatization
    p_line = stem_lem(p_line)

    return p_line

#-----------------------------------------------------------------
# Preprocesses all the sentences in a file 
#-----------------------------------------------------------------
def preprocess_file(file_name):
    f = open(file_name, 'r')

    # We had to use this because there are double interrogation questions sometimes, 
    # and nltk.sent_tokenize() would split them into different lines
    f_lines = f.read().split('\n')

    processed_lines = []

    for line in f_lines:
        processed_lines += [preprocess_line(line),]
    
    return

#--------------------------------------------------------------------------------------------
# Function that splits file_name.txt into file_name-questions.txt and file_name-labels.txt
#--------------------------------------------------------------------------------------------
def split_file(file_name):
    dev_set = open('{}.txt'.format(file_name), 'r')
    dev_set_lines = dev_set.readlines()
    dev_set.close()

    with open('{}-questions.txt'.format(file_name), 'w') as questions, open('{}-labels.txt'.format(file_name), 'w') as labels:
        for line in dev_set_lines:
            split_line = line.split(' ', 1)
            labels.write(split_line[0]+'\n')
            questions.write(split_line[1])

    return

#--------------------------------------------------------------------------------------------
# Function that reads our Training Data
#--------------------------------------------------------------------------------------------
def read_training_data(file_name):
    train_set = open('{}'.format(file_name), 'r')
    train_set_lines = train_set.readlines()
    train_set.close()

    for line in train_set_lines:
        split_line = line.split(' ', 1)
        label = split_line[0].split(':')
        phrase = split_line[1].strip()
        
        # Setting up our training data dictory to store our results
        global training_data
        global index_training_data
        if label[0] not in training_data:
            training_data[label[0]] = {}
            index_training_data[label[0]] = {}
            # create tf-idf
            index_training_data[label[0]]['tf-idf'] = None
         
        if label[1] not in training_data[label[0]]:
            training_data[label[0]][label[1]] = []
            # create tf-idf
            index_training_data[label[0]][label[1]] = None
            

        # TODO TODO TODO: Process the line and retrive information
        
        p_line = phrase

        training_data[label[0]][label[1]] += [p_line, ]

    print(index_training_data)
    return


#--------------------------
# Project main function
#--------------------------
def main():
    case = sys.argv[1]

    if case == '-setup' or case == '-coarse' or case == '-fine':
        file_name = sys.argv[2]
        if case == '-setup':
            split_file(file_name)

        elif case == '-coarse' or case == '-fine':
            dev_set_name = sys.argv[3]
            read_training_data(file_name)

            if case == '-coarse':
                return

            elif case == '-fine':
                return

    elif case == '-test':
        preprocess_file('DEV-questions.txt')

    elif case == '-help':
        help_menu()
    else:
        print("Invalid Input")

    return

#--------------------------
# Help Trace
#--------------------------

def help_menu():
    print('Help trace for program qc.py')
    print('---')
    print('Usage Commands:')
    print('-setup \'name\': Splits a data \'name\'.txt on the current directory into \'name\'-questions.txt and \'name\'-labels.txt')
    return 

main()