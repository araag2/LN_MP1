import os
import re
import sys
import copy
import statistics
import sklearn as skl
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk

#-----------------------------------------------------------------
# Global Variable Field 
#-----------------------------------------------------------------
stop_words_coarse = ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'are', 'aren', "aren't", 'as', 'at', 'be',
                     'because', 'been', 'before', 'below', 'both', 'but', 'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', "doesn't", 
                     'don', "don't", 'few', 'for', 'from', 'further', "hadn't", 'has', 'hasn', "hasn't", "haven't", 'he', 'her', 'hers', 'herself', 'his', 
                     'i', 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 
                     'mustn', "mustn't", 'needn', "needn't", 'no', 'now', 'o', 'of', 'off', 'on', 'once', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 
                     'over', 're', 's', 'same', "shan't", 'she', "she's", 'should', "should've", "shouldn't", 'so', 't', 'than', "that'll", 'the', 'their', 
                     'theirs', 'them', 'then', 'these', 'through', 'to', 'under', 'until', 've', 'very', "wasn't", 'we', 'weren', "weren't", 'what', "won't", 
                     'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']


stop_words_fine = ['above', 'ain', 'aren', "aren't", 'below', 'couldn', "couldn't", "didn't", "doesn't", "don't", 'few', 'hadn', "hadn't", 'hasn', "hasn't", 
                   "haven't", 'hers', 'herself', "isn't", "it's", 'll', 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'ours', 'ourselves', 
                   're', "shan't", "she's", "should've", "shouldn't", "that'll", 'theirs', 've', "wasn't", 'weren', "weren't", "won't", 'wouldn', "wouldn't", 
                   'y', "you'd", "you'll", "you're", "you've", 'yours', 'yourself', 'yourselves']

coarse = True

train_vectorizer = None
svm_classifier = None
vec = None
labels = None
#-----------------------------------------------------------------
# Stemming and/or Lemmatization of a line 
#-----------------------------------------------------------------
def stem_lem(line):
    global coarse

    p_line = None
    if coarse:
        lemmer = nltk.WordNetLemmatizer()
        p_line = [lemmer.lemmatize(word) for word in line]

    else:
        stemmer = nltk.stem.snowball.EnglishStemmer()
        p_line = [stemmer.stem(word) for word in line]

    return p_line

#-----------------------------------------------------------------
# Preprocesses a line 
#-----------------------------------------------------------------
def preprocess_line(line):
    global coarse

    # Lowercasing the entire line
    p_line = line.lower()
    
    if coarse:
        for word in nltk.word_tokenize(p_line):
            if word in stop_words_coarse:
                p_line = re.sub('{}\s'.format(word),'', p_line)
    else:
        for word in nltk.word_tokenize(p_line):
            if word in  stop_words_fine:
                p_line = re.sub('{}\s'.format(word),'', p_line)

    # Remove punctuation using regex
    p_line = re.sub('[?!/\.,;:`\']','',  p_line)

    # Tokenization 
    p_line = nltk.word_tokenize(p_line)

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
# Auxiliary function
#--------------------------------------------------------------------------------------------
def flatten_list(l):
    new_list = ''
    for word in l:
        new_list += ('{} '.format(word))
    return new_list.strip()

#--------------------------------------------------------------------------------------------
# Function that reads our Training Data
#--------------------------------------------------------------------------------------------
def read_train_data(file_name):
    global coarse

    train_set = open('{}'.format(file_name), 'r')
    train_set_lines = train_set.readlines()
    train_set.close()

    p_train_set = []
    train_set_labels = []

    for line in train_set_lines:
        split_line = line.split(' ', 1)
        label = split_line[0]

        if coarse:
            label = split_line[0].split(':')[0].strip()
            
        phrase = split_line[1].strip()


        p_line = preprocess_line(phrase)
        p_train_set += [flatten_list(p_line),]
        train_set_labels += [label,]

    global train_vectorizer
    global svm_classifier
    global vec
    global labels

    train_vectorizer = TfidfVectorizer()
    vec_data = train_vectorizer.fit_transform(p_train_set)

    if coarse:
        svm_classifier = svm.SVC(kernel='rbf', C=100.0, gamma=1.0)
    else:
        svm_classifier = svm.SVC(kernel='rbf', C=100.0, gamma=0.2)

    svm_classifier.fit(vec_data, train_set_labels)

    vec = vec_data
    labels = train_set_labels

#--------------------------------------------------------------------------------------------
# Function that generates course labels for each document
#--------------------------------------------------------------------------------------------

def generate_c_label(file_name):
    dev_set = open('{}'.format(file_name), 'r')
    dev_set_lines = dev_set.readlines()
    dev_set.close()

    global train_vectorizer
    global svm_classifier

    text_output = ''
    for line in dev_set_lines:
        p_line = [flatten_list(preprocess_line(line.strip()))]

        test_array = train_vectorizer.transform(p_line)
        prediction = str(svm_classifier.predict(test_array))
        text_output += re.sub('[\[\'\]]','', prediction).rstrip() + '\n'

    print(text_output.rstrip())
    return
#--------------------------
# Project main function
#--------------------------
def main():
    case = sys.argv[1]

    global stop_words_coarse
    global stop_words_fine
    global stop_words_coarse_remove
    global stop_words_fine_remove
    global stop_words_appered
    global coarse

    if case == '-setup' or case == '-coarse' or case == '-fine':
        file_name = sys.argv[2]
        if case == '-setup':
            split_file(file_name)

        elif case == '-coarse' or case == '-fine':
            dev_set_name = sys.argv[3]
            
            if case == '-coarse':
                read_train_data(file_name)

            elif case == '-fine':
                coarse = False
                read_train_data(file_name)
            
            generate_c_label(dev_set_name)

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
