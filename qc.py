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
'''
NLTK StopWordList
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
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

Words that never appear:
['ours', 'ourselves', "you're", "you've", "you'll", "you'd", 'yours', 'yourself', 'yourselves', "she's", 'hers', 'herself', "it's", 
'theirs', "that'll", 'above', 'below', 'few', "don't", "should've", 'll', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
"didn't", "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", "haven't", "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 
"shan't", "shouldn't", "wasn't", 'weren', "weren't", "won't", 'wouldn', "wouldn't"]

Words that appear:
['where', 'can', 'be', 'what', 'do', 'most', 'is', 'the', 'of', 'for', 'who', 'when', 'did', 'does', 'in', 'to', 'their', 'before', 'how', 
'a', 'on', 'it', 'why', 'we', 'by', 'an', 'was', 'from', 'her', 'are', 'i', 'you', 'which', 'and', 'has', 'all', 'he', 'during', 'were', 'if', 
'at', 'have', 'having', 'as', 'not', 'will', 'between', 'its', 'there', 'had', 'more', 'that', 's', 'no', 'should', 'or', 'only', 'himself', 'into', 
'his', 'each', 'with', 'after', 'so', 'doing', 'now', 'they', 'about', 'your', 'other', 'over', 'under', 'up', 'such', 'don', 'my', 'me', 'won', 'through', 
'out', 'some', 'just', 'been', 'own', 'am', 'than', 'those', 'because', 'them', 'this', 'our', 'itself', 'these', 'then', 'but', 'while', 'same', 'whom', 'being', 
'down', 'shan', 'here', 'him', 'off', 'd', 'shouldn', 'she', 'against', 'myself', 'any', 'again', 'very', 't', 'once', 'too', 'themselves', 'wasn', 'nor', 'doesn', 
'o', 'm', 'haven', 'both', 'didn', 'until', 'further', 'isn']
'''

stop_words_coarse = ['i', 'me', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
"you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'his', 'she', "she's", 
'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'them', 'their', 'theirs', 'what', 
"that'll", 'these', 'am', 'is', 'are', 'be', 'been', 
'has', 'do', 'did', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
'because', 'as', 'until', 'of', 'at', 'by', 'for', 'about', 'against', 'into', 'through', 
'before', 'after', 'above', 'below', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 
'again', 'further', 'then', 'once', 'all', 'both', 'few', 
'more', 'other', 'no', 'same', 'so', 'than', 'very', 's', 't', 
'can', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 
"aren't", 'couldn', "couldn't", 'didn', "didn't", "doesn't", "hadn't", 'hasn', "hasn't", "haven't", 
"isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", "shan't", "shouldn't", 
"wasn't", 'weren', "weren't", "won't", 'wouldn', "wouldn't"]

stop_words_fine = []

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
# Auxiliary function
#--------------------------------------------------------------------------------------------
def flatten_list(l):
    new_list = ''
    for word in l:
        new_list += ('{} '.format(word))
    return new_list.strip()

#--------------------------------------------------------------------------------------------
# Function that creates an empty TF-IDF index
#--------------------------------------------------------------------------------------------
def create_index():
    return TfidfVectorizer(use_idf=True)

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

    # Best so far: Coarse [SVC(C=100.0, gamma=1)]
    # 82.14080% Coarse [SVC(C=70.0, gamma=0.5)]
    # Best so far: Fine [SVC(C=100.0, gamma=0.1)]
    # 75.81% [SVC(C=20, gamma=0.4)]
    # 78.023134% [SVC(C=100, gamma=0.2)]

    if coarse:
        svm_classifier = svm.SVC(kernel='rbf', C=100.0, gamma=1)
    else:
        svm_classifier = svm.SVC(kernel='rbf', C=100.0, gamma=0.2)

    svm_classifier.fit(vec_data, train_set_labels)

    vec = vec_data
    labels = train_set_labels

#--------------------------------------------------------------------------------------------
# Auxiliary function just to test classifier values
#--------------------------------------------------------------------------------------------
#TODO: Remove this afterwards
def test_classifier_values(dev_set_lines):
    global vec
    global labels

    classifiers = []
    for C in range(70,101,1):
        for gamma in range(1,11,1):
            classifiers += [[C, gamma/10], ]
            print(C, gamma/10)


    sol = open('DEV-labels.txt', 'r')
    temp_sol_lines = sol.readlines()
    sol.close()

    sol_lines = []

    for line in temp_sol_lines:
        p_line = re.sub('[\s\x00\\n]','',line)
        if p_line != '':
            sol_lines += [p_line,]

    best_accuracy = 0
    best_accuracy_measures = None

    for classifier in classifiers:
        out_lines = []
        n_labels = 0
        correct_labels = 0
        clf = svm.SVC(C=classifier[0], gamma=classifier[1])
        clf.fit(vec, labels)

        for line in dev_set_lines:
            p_line = [flatten_list(preprocess_line(line.strip()))]
            test_array = train_vectorizer.transform(p_line)
            prediction = str(clf.predict(test_array))
            out_lines += [re.sub('[\[\'\]]','', prediction).rstrip(),]

        for i in range(len(sol_lines)):
            if re.sub('[\s\x00]','',out_lines[i]) in str(sol_lines[i]):
                correct_labels += 1
            n_labels += 1 
        accuracy = (correct_labels/n_labels)*100
        print("Current accuracy {:3f}%".format(accuracy))
        print("Current measures {}".format([clf]))

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_accuracy_measures = [clf]

    print("Best accuracy {:3f}%".format(best_accuracy))
    print("Best measures {}".format(best_accuracy_measures))

    return
#--------------------------------------------------------------------------------------------
# Function that generates course labels for each document
#--------------------------------------------------------------------------------------------

def generate_c_label(file_name):
    dev_set = open('{}'.format(file_name), 'r')
    dev_set_lines = dev_set.readlines()
    dev_set.close()

    global train_vectorizer
    global svm_classifier

    #TODO: Remove this afterwards
    #test_classifier_values(dev_set_lines)

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