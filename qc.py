import os
import re
import sys
import sklearn as skl
import numpy as np
import nltk
from nltk.corpus import stopwords

# Variables
stop_words = []

training_data = {}

#-----------------------------------------------------------------
# Creates our stop words list
#-----------------------------------------------------------------

def create_stop_words():
    # If necessary uncoment this
    # nltk.download('stopwords')
    nltk_stop_words = stopwords.words('english')

    # TODO: Check How/What/Why/When because they convey meaning
    global stop_words 
    stop_words = nltk_stop_words

    return stop_words

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

    print(p_line)
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
        
        # TODO TODO TODO: Process the line and retrive information
        p_line = phrase

        global training_data
        if label[0] not in training_data:
            training_data[label[0]] = {} 
         
        if label[1] not in training_data[label[0]]:
            training_data[label[0]][label[1]] = [] 

        training_data[label[0]][label[1]] += [p_line, ]

    return

#--------------------------
# Project main function
#--------------------------
def main():
    case = sys.argv[1]
    file_name = sys.argv[2]

    create_stop_words()

    if case == '-setup':
        split_file(file_name)
    elif case == '-coarse' or case == '-fine':
        read_training_data(file_name)

    elif case == '-test':
        preprocess_file('DEV-questions.txt')
    else:
        print("Invalid Input")

    return

main()