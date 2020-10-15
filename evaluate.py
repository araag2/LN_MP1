import sys
import os
import re
import nltk
#--------------------------
# Evaluating files
#--------------------------
def evaluate_files(solution, output):
    sol = open(solution, 'r')
    out = open(output, 'r')

    temp_sol_lines = sol.readlines()
    temp_out_lines = out.readlines()

    sol.close()
    out.close()

    sol_lines = []
    out_lines = []

    for line in temp_sol_lines:
        p_line = re.sub('[\s\x00\\n]','',line)
        if p_line != '':
            sol_lines += [p_line,]

    for line in temp_out_lines:
        p_line = re.sub('[\s\x00\\n]','',line)
        if p_line != '':
            out_lines += [p_line,]

    correct_labels = 0
    n_labels = 0       
    
    print(sol_lines)
    print(out_lines)
    print('The len of our solution set is {}'.format(len(sol_lines)))
    print('The len of our output set is {}'.format(len(out_lines)))

    for j in range(len(sol_lines)):
        if out_lines[j] == '\n':
            continue

        if re.sub('[\s\x00]','',out_lines[j]) in str(sol_lines[j]):
            correct_labels += 1
        n_labels +=1 

    accuracy = (correct_labels/n_labels)*100

    print('Baseline accuracy is {:.3f}'.format(accuracy))

    return 

#--------------------------
# Project main function
#--------------------------
def main():
    solution_file = sys.argv[1]
    output_file = sys.argv[2]

    evaluate_files(solution_file, output_file)
    return

main()