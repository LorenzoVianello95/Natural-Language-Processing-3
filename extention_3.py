
#HOMEWORK 3 ESTENTION1.3


import numpy as np
import sys
from string import punctuation
import pickle
import string
import operator

dataset_rep="SRLData/EN/"
dev_txt="CoNLL2009-ST-English-development.txt"
train_txt="CoNLL2009-ST-English-train.txt"
import os
from gensim.models import Word2Vec
from nltk.corpus import wordnet as wn
from nltk.corpus import propbank as pb


#Function that reads the dataset conl, for each word it extracts lemma and pos
stopwords = set([w.rstrip('\r\n') for w in open("StopwordsM.txt")])

def read_data(txt_file):
    in_file = open(txt_file,"r")
    all_text=[]
    sentence=[]
    w2v_text_lemmas=[]#w2v model for only lemmas
    w2v_sent_lemmas=[]
    w2v_text_sense=[]#w2v model for lemmas and sense
    w2v_sent_sense=[]
    for line in in_file:
        line.replace("\n","")
        split=line.strip("\n").split("\t")
        if len(split)==1:
            if not(len(sentence)==0):
                all_text.append(sentence)
            sentence=[]
            if not(len(w2v_sent_lemmas)==0):
                w2v_text_lemmas.append(w2v_sent_lemmas)
            w2v_sent_lemmas=[]
            if not(len(w2v_sent_sense)==0):
                w2v_text_sense.append(w2v_sent_sense)
            w2v_sent_sense=[]
        else:
            for k in [0, 0, 1, 2, 2, 2, 2, 2, 3]:
                #print(split)
                del split[k]
            #sentence.append(split)
            if not split[0] in punctuation: #COS ESCLUDO SOLO LE STOPWORD,
                sentence.append(split)
                w2v_sent_lemmas.append(split[0])
                if split[4] == "_":
                    w2v_sent_sense.append(split[0])
                else:
                    w2v_sent_sense.append(split[4])
    in_file.close()
    return all_text,w2v_text_lemmas,w2v_text_sense

train_matrix, w2v_lemmas, w2v_sense= read_data(dataset_rep+train_txt)


#For each predicate I extract the args from the newly calculated matrix and sort everything into a dictionary of this type:
#{pred:{arg1:[....],arg2:[....]...},pred2:....}
def calculate_args(matrix):
    dict_pred_lemmas={}
    dict_pred_args={}
    for sent in matrix:#[0:10000]:
        d_np={}
        d_pl={}
        d_na={}
        counter=0
        for word in sent:
            if not word[4]=="_":
                d_np[counter]=word[4]
                d_pl[word[4]]=word[0]
                counter+=1
            if len(word)>5:
                for l in range(5,len(word)):
                    if not word[l]=="_":
                        if l-5 in d_na.keys():
                            mom=d_na[l-5]

                            if word[l] in mom.keys():
                                #print(d_na[l - 5])
                                mom1= mom[word[l]]
                                mom1.append([word[0],word[1]])
                                mom[word[l]]=mom1
                                d_na[l - 5] = mom
                                #print(d_na[l - 5])

                            else:
                                mom[word[l]]=[[word[0],word[1]]]
                                d_na[l - 5]=mom
                        else:
                            d_na[l-5]={word[l]:[[word[0],word[1]]]}
        for k in d_pl.keys():
            if not k in dict_pred_lemmas.keys():
                dict_pred_lemmas[k]=d_pl[k]

        for n in d_np.keys():
            if d_np[n] in dict_pred_args.keys():
                mom=dict_pred_args[d_np[n]]
                if n in d_na.keys():
                    for k in d_na[n].keys():
                        if k in mom.keys():
                            mom1=mom[k]
                            mom2=d_na[n][k]
                            mom1=mom1+mom2
                            mom[k]=mom1
                        else:
                            mom[k]=d_na[n][k]
                dict_pred_args[d_np[n]]=mom
            else:
                if n in d_na.keys():
                    dict_pred_args[d_np[n]]=d_na[n]

    return dict_pred_lemmas,dict_pred_args

dict_pred_lemmas, dict_pred_args=calculate_args(train_matrix)

#this method is my attempt to associate the predicates present in the vocabulary described above with those of PropBank,
#As input, it receives an element of the dictionary that corresponds to a predicate
#As output it returns an attempt to describe the arguments
def from_conl_to_propBank(predicato):
    #try:
        print(predicato)
        eat_01 = pb.roleset(predicato)
        for role in eat_01.findall("roles/role"):
            #print(role)
            descr= role.attrib['descr']
            print("     "+descr)
            descr=descr.translate(None, string.punctuation)
            split=descr.strip().split()
            dict_max_arg={}
            for k in dict_pred_args[predicato].keys():
                mean=0
                count=0
                args= dict_pred_args[predicato][k]
                for element in args:
                    #print(element)
                    #sys.exit("k")
                    #print(element)
                    p_o_s=element[1]
                    element = element[0]
                    if p_o_s.startswith("NN"):
                        e = wn.synsets(element,pos="n")
                    elif p_o_s.startswith("VB"):
                        e = wn.synsets(element, pos="v")
                    else:
                        e= wn.synsets(element)
                    if len(e)>0:
                        for s in split:
                            ss = wn.synsets(s)
                            if len(ss) > 0:
                                sim= e[0].path_similarity(ss[0])
                                if sim>0:
                                    #print(e[0],ss[0])
                                    count+=1
                                    mean+=sim
                if count>0:
                    mean=mean/count
                dict_max_arg[k]= mean
            arg_max=max(dict_max_arg.iteritems(), key=operator.itemgetter(1))[0]
            print("         "+str(arg_max)+ str(dict_pred_args[predicato][arg_max]) )

#this is an example of the input
from_conl_to_propBank("drink.01")

# the output look like this:
#drink.01
#     drinker
#         A0[['you', 'PRP'], ['i', 'PRP'], ['me', 'PRP'], ['people', 'NN'], ['gemsbok', 'NNS'], ['sindona', 'NNP'], ['americans', 'JJ'], ['i', 'PRP'], ['texans', 'JJ'], ['lots', 'NNS']]
#     liquid
#         A1[['more', 'DT'], ['more', 'DT'], ['tea', 'NN'], ['water', 'NN'], ['coffee', 'NN'], ['less', 'DT'], ['tea', 'NN'], ['beer', 'NN']]


sys.exit("k")
#this second part contain some attempts I made, some results are
#presented inside the report

#calcola centroid in base a w2v
def extract(tree):
    model = Word2Vec(w2v_lemmas, size=100, window=5, min_count=1, workers=4)
    model.train(w2v_lemmas, total_examples=len(w2v_lemmas), epochs=10)
    word_vectors = model.wv
    new_tree={}
    for k in tree.keys():
        word=tree[k]
        new_sub={}
        for k1 in word.keys():
            args=word[k1]
            centroid=np.zeros(100)
            c=0
            for el in args:
                if el in word_vectors.vocab:
                    mm=model[el]
                    centroid+= mm
                    c+=1
            centroid/=c
            nw = model.most_similar( [ centroid ], [], 1)
            new_sub[k1]=nw[0][0]
        new_tree[k]=new_sub
    return new_tree


a=["pizza","icecream","bowl","rice", "apples","arts"]

def find_bests_Hyp(a):
    d={}
    d1={}
    #calcolo i pesi che hanno le singole parole rispetto alle altre in base alla similarity (considero solo il primo significato)
    for el in a:
        l = wn.synsets(el)
        if len(l) > 0:
            e = l[0]
            counter=0
            for el1 in a:
                l1 = wn.synsets(el1)
                if not(el1==el) and len(l1)>0:
                    #print el1,el
                    e1=l1[0]
                    sim=e.path_similarity(e1)
                    if sim>0:
                        counter+=sim
            d1[el]=counter

    #calcolo gli hypernyms piu' comuni e li ordino in base ai pesi delle parole
    for el in a:
        l=wn.synsets(el)
        if len(l)>0:
            e = l[0]
            add=d1[el]
            s=set([i for i in e.closure(lambda s:s.hypernyms())])
            for g in s:
                    if g in d.keys():
                            f=d[g]
                            d[g]=f+add
                    else:
                            d[g]=add
    return  sorted(d.items(), key= operator.itemgetter(1))
d=find_bests_Hyp(a)
print(d)

#find the WordNet Lowest Common Hypernym
def find_Lowest_Common_Hypernym(a):
    for element in a:
        l = wn.synsets(element)
        if len(l)>0:
            ch = l[0]
            for el in a:
                l1= wn.synsets(el)
                if len(l1) > 0:
                    ch1=l1[0]
                    lch=ch.lowest_common_hypernyms(ch1)
                    #print lch
                    if len(lch)>0:
                        ch=lch[0]
            return ch

print(find_Lowest_Common_Hypernym(a))
