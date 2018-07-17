import numpy as np
import sys
from string import punctuation
import pickle

dataset_rep="SRLData/EN/"
dev_txt="CoNLL2009-ST-English-development.txt"
train_txt="CoNLL2009-ST-English-train.txt"
test_txt="SRLData/TestData/test.csv"
import os
from gensim.models import Word2Vec

flag_save=1

#return:
#all dataset
#[
#   [ sentence1
#       [word1 with all its features]
#       [word2..]
#       ...
#   ]
#   [sentence2]
#   ...
#]
#input words have this format:
#[ID FORM LEMMA PLEMMA POS PPOS FEAT PFEAT HEAD PHEAD DEPREL PDEPREL FILLPRED PRED APREDs ]
#what i need for my processing are these parts:
#[ LEMMA  POS  DEPREL  FILLPRED PRED APREDs ]

#output contain also the sentences for at the end build w2v embedding
#I eliminate some stop words
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
dev_matrix,_,_= read_data(dataset_rep+dev_txt)
test_matrix,_,_= read_data(test_txt)

#following the advice of the paper as input, I choose to represent the words as:
#[x1,x2,x3,x4,flag]
#where x1  is the value associated to the lemma
#x2 is the value associated with the pos
#x3 is the value associated with the deprel
#x4 is the value associated with the predicates disambiguated
#flag is a flag that tell if it is or not [0 or 1] the pred we are considering

#with this function I build the vocabularies
#to better visualize the results I build 5 vocabularies, one for each x and 1 for the labels
#the vocabularies I build are based on the train set only so if a word is not present
#there is the "UNK" simbol;
#only vocabularies for lemmas and preds have max length, inside them there are only the words more frequentes
def build_vocabolaries(matrix,lemmas_vocab_size,pred_vocab_size):
    voc_lemmas={"UNK":0, "_":1}
    freq_dict_lemmas={}
    voc_pos={"UNK":0, "_":1}
    voc_deprel={"UNK":0, "_":1}
    voc_pred={"UNK":0, "_":1}
    freq_dict_pred={}
    voc_apreds={"UNK":0, "_":0}#scelgo di mettere unk e _ entrambi a zero
    ps=2
    d=2
    a=1
    for j in range(len(matrix)):
        for i in range(len(matrix[j])):
            word=matrix[j][i]
            labels_numb_word = []

            if word[0] in freq_dict_lemmas.keys():
                c = freq_dict_lemmas.get(word[0])
                freq_dict_lemmas[word[0]] = c + 1
            else:
                freq_dict_lemmas[word[0]] =1
            if not(word[1] in voc_pos.keys()):
                voc_pos[word[1]]=ps
                ps+=1
            if not(word[2] in voc_deprel.keys()):
                voc_deprel[word[2]]=d
                d+=1
            if word[4] in freq_dict_pred.keys():
                c = freq_dict_pred.get(word[4])
                freq_dict_pred[word[4]] = c + 1
            elif not word[4]=="_":
                freq_dict_pred[word[4]] =1
            if len(word)>5:
                for index in range(5,len(word)):

                    if not (word[index] in voc_apreds.keys()):
                        voc_apreds[word[index]] = a
                        a += 1
                    labels_numb_word.append(voc_apreds[word[index]])

    listW_LEMMAS = sorted(freq_dict_lemmas, key=freq_dict_lemmas.get, reverse=True)
    for index in range(2, lemmas_vocab_size):
        if listW_LEMMAS:
            mom = listW_LEMMAS.pop(0)
            voc_lemmas[mom] = index
    listW_PRED = sorted(freq_dict_pred, key=freq_dict_pred.get, reverse=True)
    for index in range(2, pred_vocab_size):
        if listW_PRED:
            mom = listW_PRED.pop(0)
            voc_pred[mom] = index
    return voc_lemmas,voc_pos,voc_deprel,voc_pred,voc_apreds#,input_numb_matrix,labels_numb_matrix

voc_lemmas, voc_pos, voc_deprel, voc_pred,voc_apreds =build_vocabolaries(train_matrix,7000,3000)

dimensions_1hotencoding=[len(voc_lemmas),len(voc_pos),len(voc_deprel),len(voc_pred),len(voc_apreds)]

#I choose to save some information that I need also the NN
if flag_save==1:
    if os.path.isfile('voc_lemmas.pckl'):
        os.remove('voc_lemmas.pckl')
    f = open('voc_lemmas.pckl', 'wb')
    pickle.dump(voc_lemmas, f)
    f.close()

    if os.path.isfile('voc_apreds.pckl'):
        os.remove('voc_apreds.pckl')
    f = open('voc_apreds.pckl', 'wb')
    pickle.dump(voc_apreds, f)
    f.close()

    if os.path.isfile('1hotencoding.pckl'):
        os.remove('1hotencoding.pckl')
    f = open('1hotencoding.pckl', 'wb')
    pickle.dump(dimensions_1hotencoding, f)
    f.close()

# using the vocabularies already created I transform the matrix from lemma to numbers
def transform_w2v(matrix):
    input_numb_matrix=[]
    labels_numb_matrix=[]
    for i in range(len(matrix)):
        input_sent = []
        labels_sent=[]
        counter_predicates=1
        for j in range(len(matrix[i])):
            word=matrix[i][j]
            labels_numb_word = []
            input_numb_word =[]

            if word[0] in voc_lemmas.keys():
                input_numb_word.append(voc_lemmas[word[0]])
            else:
                input_numb_word.append(voc_lemmas["UNK"])

            if word[1] in voc_pos.keys():
                input_numb_word.append(voc_pos[word[1]])
            else:
                input_numb_word.append(voc_pos["UNK"])

            if word[2] in voc_deprel.keys():
                input_numb_word.append(voc_deprel[word[2]])
            else:
                input_numb_word.append(voc_deprel["UNK"])

            if word[4] in voc_pred.keys():
                input_numb_word.append(voc_pred[word[4]])
            else:
                input_numb_word.append(voc_pred["UNK"])

            if len(word) > 5:
                for index in range(5, len(word)):
                    if word[index] in voc_apreds.keys():
                        labels_numb_word.append(voc_apreds[word[index]])
                    else:
                        labels_numb_word.append(voc_apreds["UNK"])
            flag = 0
            if not input_numb_word[3] == 1:
                flag = counter_predicates
                counter_predicates += 1
            input_numb_word.append(flag)

            input_sent.append(input_numb_word)
            labels_sent.append(labels_numb_word)
        input_numb_matrix.append(input_sent)
        labels_numb_matrix.append(labels_sent)
    return input_numb_matrix, labels_numb_matrix

input_matrix, labels_matrix= transform_w2v(train_matrix)
dev_input_matrix, dev_labels_matrix=transform_w2v(dev_matrix)
test_input_matrix, _=transform_w2v(test_matrix)

if flag_save==1:
    if os.path.isfile('testset.pckl'):
        os.remove('testset.pckl')
    f = open('testset.pckl', 'wb')
    pickle.dump(test_input_matrix, f)
    f.close()


# now I need to divide the labels each column of them are associated to a predicate,
# so I build how many sentences as the number of pred, I also put the flag information inside the words representation
def divide_labels(matrix_inputs,matrix_labels):
    new_matrix_inputs=[]
    new_matrix_labels = []
    for i in range(len(matrix_inputs)):
        for p in range(len(matrix_labels[i][0])):
            input_sent = []
            labels_sent = []
            for j in range(len(matrix_inputs[i])):
                word_input = matrix_inputs[i][j]
                word_labels= matrix_labels[i][j]
                if word_input[4]==0:
                    input_sent.append(word_input)
                elif word_input[4]==p+1:
                    new_word_input=[word_input[k] for k in range(4)]
                    new_word_input.append(1)
                    input_sent.append(new_word_input)
                else:
                    new_word_input = [word_input[k] for k in range(4)]
                    new_word_input.append(0)
                    input_sent.append(new_word_input)

                labels_sent.append(word_labels[p])
            new_matrix_inputs.append(input_sent)
            new_matrix_labels.append(labels_sent)
    return new_matrix_inputs,new_matrix_labels

dev_input_matrix, dev_labels_matrix=divide_labels(dev_input_matrix, dev_labels_matrix)
input_matrix, labels_matrix=divide_labels(input_matrix, labels_matrix)

#now we build the w2v embeeding using gensim:
# There are 2 embeeding; 1 for lemmas so the input sentences are like:
#       [[we,love,pizza],[i,hate,icecream]...]
# and 1 for predicates , for this we have sentences like:
#       [[we,love.01, pizza][i,hate.01, icecream]...]
def build_w2v_embedding(sentences, flag):
    if flag=="lemmas":
        new_sentences=[["_","_","_","_","_","_"]]
        for s in sentences:
            new_s=[]
            for w in s:
                if w in voc_lemmas.keys():
                    new_s.append(w)
                else:
                    new_s.append("UNK")
            new_sentences.append(new_s)
        model = Word2Vec(new_sentences, size=100, window=5, min_count=5, workers=4)
        listW = sorted(voc_lemmas, key=voc_lemmas.get, reverse=False)
        embedding=[]
        for w in listW:
            print(w,listW.index(w))
            embedding.append(model[w])
    if flag=="senses":
        new_sentences=[["_","_","_","_","_","_"]]
        for s in sentences:
            new_s=[]
            for w in s:
                if w in voc_pred.keys():
                    new_s.append(w)
                elif w in voc_lemmas.keys():
                    new_s.append(w)
                else:
                    new_s.append("UNK")
            new_sentences.append(new_s)
        model = Word2Vec(new_sentences, size=100, window=5, min_count=5, workers=4)
        listW = sorted(voc_pred, key=voc_pred.get, reverse=False)
        embedding=[]
        for w in listW:
            print(w,listW.index(w))
            embedding.append(model[w])
    return embedding

lemmas_EMBEDDING= build_w2v_embedding(w2v_lemmas,"lemmas")
senses_EMBEDDING= build_w2v_embedding(w2v_sense,"senses")

#all the data I created are saved to be used in the next passage
if flag_save==1:
    print("preprocessing concluso salvataggio dati")

    if os.path.isfile('X.pckl'):
        os.remove('X.pckl')
    f = open('X.pckl', 'wb')
    pickle.dump(input_matrix, f)
    f.close()
    if os.path.isfile('Y.pckl'):
        os.remove('Y.pckl')
    f = open('Y.pckl', 'wb')
    pickle.dump(labels_matrix, f)
    f.close()
    if os.path.isfile('X_dev.pckl'):
        os.remove('X_dev.pckl')
    f = open('X_dev.pckl', 'wb')
    pickle.dump(dev_input_matrix, f)
    f.close()
    if os.path.isfile('Y_dev.pckl'):
        os.remove('Y_dev.pckl')
    f = open('Y_dev.pckl', 'wb')
    pickle.dump(dev_labels_matrix, f)
    f.close()

    if os.path.isfile('lemmas_embedding.pckl'):
        os.remove('lemmas_embedding.pckl')
    f = open('lemmas_embedding.pckl', 'wb')
    pickle.dump(lemmas_EMBEDDING, f)
    f.close()
    if os.path.isfile('senses_embedding.pckl'):
        os.remove('senses_embedding.pckl')
    f = open('senses_embedding.pckl', 'wb')
    pickle.dump(senses_EMBEDDING, f)
    f.close()



