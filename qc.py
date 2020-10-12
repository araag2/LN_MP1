import os
import re
import sys
import sklearn as skl
import numpy as np
import nltk

#-----------------------------------------------------------------
# Preprocesses a line 
#-----------------------------------------------------------------
def preprocess_line(line):
    # Lowercasing the entire line
    processed_line = line.lower()
    # Remove punctuation using regex
    processed_line = re.sub('[?!\.,;:`\']','',processed_line)


    return processed_line

#-----------------------------------------------------------------
# Preprocesses all the sentences in a file 
#-----------------------------------------------------------------
def preprocess_file(file_name):
    f = open(file_name, 'r')
    f_lines = nltk.sent_tokenize(f.read())
    processed_lines = []

    for line in f_lines:
        processed_lines += [preprocess_line(line),]
    
    print(processed_lines)
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

#--------------------------
# Project main function
#--------------------------
def main():
    case = sys.argv[1]

    if case == '-setup':
        split_file('DEV')
    elif case == '-coarse':
        return
    elif case == '-fine':
        return
    elif case == '-test':
        preprocess_file('DEV-questions.txt')
    else:
        print("Invalid Input")

    return

main()