import sys
import os
import re
#--------------------------
# Evaluating files
#--------------------------
def evaluate_files(filename1, filename2):
    f1 = open(filename1, 'r')
    f2 = open(filename2, 'r')

    f1_lines = f1.readlines()
    f2_lines = f2.readlines()

    f1.close()
    f2.close()

    count = 0
    i = 0       
    j = 0
    
    for j in range(len(f1_lines)):
        if re.sub('[\s\x00]','',f2_lines[j]) in str(f1_lines[j]):
            count += 1
        i +=1 

    return (count/i)*100

#--------------------------
# Project main function
#--------------------------
def main():
    solution_file = sys.argv[1]
    output_file = sys.argv[2]

    res = evaluate_files(solution_file, output_file)
    print('Baseline accuracy is {:.3f}'.format(res))

    return

main()