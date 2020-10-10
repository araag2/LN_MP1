import os
import re
import sys
import sklearn as skl
import numpy as np
import nltk

#-----------------------------------------------------------------------
# Function that splits DEV.txt into DEV-questions.txt and DEV-labels.txt
#-----------------------------------------------------------------------
def split_dev_file(file_name):
    dev_set = open('DEV.txt', 'r')

    ### Check this
    #dev_set_lines = nltk.sent_tokenize(dev_set.read())
    dev_set_lines = dev_set.readlines()
    dev_set.close()

    with open('DEV-questions.txt', 'w') as questions, open('DEV-labels.txt', 'w') as labels:
        for line in dev_set_lines:
            split_line = line.split(' ', 1)
            labels.write(split_line[0]+'\n')
            questions.write(split_line[1])

    return

#----------------------
# Project main function
#----------------------
def main():
    arg1 = sys.argv[1]
    if arg1 == '-setup':
        split_dev_file('DEV.txt')

main()