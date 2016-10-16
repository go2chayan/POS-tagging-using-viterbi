# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 15:19:25 2015
Homework 1: Implement the perceptron decoder using data and weights given
in /u/cs448/data/pos. What is your accuracy on the test file?

@author: Md Iftekhar Tanveer (itanveer@cs.rochester.edu)
"""
import numpy as np
import re

# Reads the emission and transition weights from file
def readwt(weightfile):
    with open(weightfile) as f:
        E = {}
        T = {}
        for aline in f:
            tag,weight = aline.strip().split(' ')
            tag = tag.split('_')
            # I am trying to avoid any mistake from upper/lower case issues
            # by using all lower case words.
            if tag[0]=='E':
                E[tag[1],tag[2].lower()]=float(weight)
            else:
                T[tag[1],tag[2]]=float(weight)
    return E,T

# reads the weights-file and creates a list of all the available tags
# it also writes the list of tags in a line named alltags
def savealltags():
    with open('./train.weights') as f:
        uniquetags = set()
        for aline in f:
            tag,weight = aline.strip().split(' ')
            tag = tag.split('_')
            uniquetags.add(tag[1])
    with open('./alltags','w') as f:
        for item in uniquetags:
            print>>f,item;
    print uniquetags

# Reads the list of tags from file
def readalltags(tagsetfile):
    with open(tagsetfile) as f:
        tags = [item.strip() for item in f]
    return tags

# Applies dynamic programming to find the best tag sequence
def viterbi(line,E,T,tags):
    wrdlist = line.split(' ')
    x = np.ones((len(tags),len(wrdlist)))*-1*np.inf
    b = np.zeros((len(tags),len(wrdlist)))
    for i,aword in enumerate(wrdlist):
        # As I didn't see any start or end tag in the tagset, I am assuming
        # all the weights for transition from the start tag to any other tag
        # is zero (which is not true in reality).
        # So for the first word, I don't consider the transition prob
        if i==0:
            for tagid,atag in enumerate(tags):
                x[tagid,i] = E.get((atag,aword.lower()),-1*np.inf)
                b[tagid,i] = -1 # Means this is the first word
            continue
                
        # if not the first word, consider both transition and emission prob
        for atagid,atag in enumerate(tags):
            # theoretically, the weights should be -ve inf if a specific
            # pair is not found in the corpus. However, something didn't
            # appear in the corpus doesn't mean that its probability is
            # totally zero. So, I am assigning a small value instead of
            # -ve inf.
            emmval = E.get((atag,aword.lower()),-1*1e10) #emission prob
            for atagid_prev,atag_prev in enumerate(tags):
                trval = T.get((atag_prev,atag),-1*1e10)  #transition prob
                total = x[atagid_prev,i-1]+emmval+trval 
                # Debug
#                print 'currtag',atag+'('+str(atagid)+')','prevtag',atag_prev+\
#                '('+str(atagid_prev)+')','i',str(i),'word',aword,\
#                'emm',emmval,'trans',trval,'tot',total                
                if total>x[atagid,i]:
                    x[atagid,i] = total  # Take the maximum logprob
                    b[atagid,i] = atagid_prev # keep a backward pointer
    idx = np.argmax(x[:,-1])
    annot=[]
    # Trace back the sequence using the back pointer
    for idx_ in xrange(np.size(b,axis=1),0,-1):
        annot.append(tags[int(idx)])
        idx = b[idx,idx_-1]
    annot.reverse()
    return wrdlist,annot

# Calculate the accuracy over a given test file
def calcaccuracy(file,E,T,tags):
    with open(file) as f:
        totalWords=0.
        countCorrect=0.
        for aline in f:
            data = [item.strip() for index, item in \
            enumerate(aline.strip().split(' ')) if not \
            (re.search('-[A-Z]+-',item) or index==0)]
            testline = ' '.join(data[0::2])
            annotGT = data[1::2]
            wrdlst,annt=viterbi(testline,E,T,tags)
            countCorrect=countCorrect+sum([a1==a2 for a1,a2 in zip(annotGT,annt)])
            totalWords=totalWords+len(annotGT)
            print 'acc=',float(countCorrect)/totalWords,'correct=',\
            countCorrect,'total=',totalWords
    return float(countCorrect)/totalWords,countCorrect,totalWords

# Main method
def main():
    E,T = readwt('./train.weights')
    tags = readalltags('./alltags')
    acc,correct,total = calcaccuracy('./test',E,T,tags)
    print 'Total Words =',total
    print 'Correctly tagged =',correct
    print 'accuracy=',acc*100,'%'

# For debug purpose
def unittest():
    E,T = readwt('./train.weights')
    tags = readalltags('./alltags')
    # acc,correct,total = calcaccuracy('./problemtestCase',E,T,tags)
    wrdlst,annt = viterbi("This is a test .",E,T,tags)
    print zip(wrdlst,annt)
            
if __name__=='__main__':
    main()