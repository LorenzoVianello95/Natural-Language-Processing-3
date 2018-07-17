import numpy as np
import sys
from string import punctuation
import pickle

dataset_rep="SRLData/EN/"
dev_txt="CoNLL2009-ST-English-development.txt"
train_txt="CoNLL2009-ST-English-train.txt"
import os
from gensim.models import Word2Vec
from nltk.tag.mapping import tagset_mapping
from utils_functions import unify_sens_lemmas_encoding, build_idtowsense, convert, bn_to_conl

#Program that preproced the data of conl and semcor to try
# to create a unique dataset with which to train my network and to increase the ability to disambiguate


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
#function similar to that used in the srl: from a configuration of the type:
#[ID FORM LEMMA PLEMMA POS PPOS FEAT PFEAT HEAD PHEAD DEPREL PDEPREL FILLPRED PRED APREDs ]
#I excract:
#[ LEMMA  POS  PRED]

#costruisco anche w2v sentences per poi costruire il modello con gensim
#in partticolare costruisco due modelli uno con solo i lemmas e uno lemmas e senses
stopwords = set([w.rstrip('\r\n') for w in open("StopwordsM.txt")])

def read_data(txt_file):
    in_file = open(txt_file,"r")
    all_text=[]
    mask=[]
    mask_sentence=[]
    sentence=[]
    w2v_text_lemmas=[]#w2v model for only lemmas
    w2v_sent_lemmas=[]
    w2v_text_sense=[]#w2v model for lemmas and sense
    w2v_sent_sense=[]
    for line in in_file:
        line.replace("\n","")
        split=line.strip("\n").split("\t")
        if len(split)==1:
            if not(len(mask_sentence)==0):
                mask.append(mask_sentence)
            mask_sentence=[]
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
            split=[split[i] for i in range(5)]
            del split[3]
            del split[2]
            #sentence.append(split)
            if not split[0] in punctuation: #COS ESCLUDO SOLO LE STOPWORD,
                if (not split[0] in stopwords) and len(split[0])>0 and (not split[0] in ["''",'``']):
                    if split[2]=="_":
                        split=[split[0], split[1], split[0]]
                        mask_sentence.append(0)
                    else:
                        mask_sentence.append(1)
                    sentence.append(split)
                    w2v_sent_lemmas.append(split[0])
                    w2v_sent_sense.append(split[2])
    in_file.close()
    return all_text,w2v_text_lemmas,w2v_text_sense,mask

# same program but for test set
def read_data_test(txt_file):
    in_file = open(txt_file,"r")
    all_text=[]
    mask=[]
    mask_sentence=[]
    sentence=[]
    w2v_text_lemmas=[]#w2v model for only lemmas
    w2v_sent_lemmas=[]
    w2v_text_sense=[]#w2v model for lemmas and sense
    w2v_sent_sense=[]
    for line in in_file:
        line.replace("\n","")
        split=line.strip("\n").split("\t")
        if len(split)==1:
            if not(len(mask_sentence)==0):
                mask.append(mask_sentence)
            mask_sentence=[]
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
            split=[split[i] for i in range(2)]
            split.append("_")
            #del split[3]
            #del split[2]
            #sentence.append(split)
            if not split[0] in punctuation: #COS ESCLUDO SOLO LE STOPWORD,
                if (not split[0] in stopwords) and len(split[0])>0 and (not split[0] in ["''",'``']):
                    if split[2]=="_":
                        split=[split[0], split[1], split[0]]
                        mask_sentence.append(0)
                    else:
                        mask_sentence.append(1)
                    sentence.append(split)
                    w2v_sent_lemmas.append(split[0])
                    w2v_sent_sense.append(split[2])
            #for i in range(3,len(split)):           #SE METTIQUELLO SOPRA METTI ANCHE L'ELSE
            #    if not(split[i]=="_"):          #COS CONSIDERO RILEVANTI SOLO LE PAROLE CHE SONO DISAMBIGUE
            #       sentence.append(split)      #OPPURE SOLO QUELLE CON UN ARGOMENTO
            #        break
            #else:
    in_file.close()
    return all_text,w2v_text_lemmas,w2v_text_sense,mask


train_matrix, w2v_lemmas, w2v_sense, mask_train= read_data(dataset_rep+train_txt)
dev_matrix,_,_,mask_dev= read_data(dataset_rep+dev_txt)
test_matrix, _, _, _= read_data_test("/home/lollo/Desktop/NLP/HW3/SRLData/TestData/testverbs.csv")

#I create an embedding that contains CONL predicates
new_conl_embedding= unify_sens_lemmas_encoding(senses=w2v_sense, lemmas=w2v_lemmas)

#PREPARE BN DATSET
# BUILD DICTIONARIES THAT LINK senseval2.d000.s002.t000  TO  bn:00030861n (for example)
d_idtws_train=build_idtowsense('/home/lollo/Desktop/NLP/Homework2/TRAIN/semcor.gold.key.bnids.txt')
#I extract the dataset from SemCor
l,p,_,_,y,_ ,voc_l_s= convert('/home/lollo/Desktop/NLP/Homework2/TRAIN/semcor.data.xml',d_idtws_train)

w2v_lemmas_2=l
w2v_senses_2=y
voc_lemmas_to_bn_sinsets=voc_l_s #vocabolario che mappa da lemma a bn sinset

#I create an embedding that contains SemCor predicates
new_bn_embedding= unify_sens_lemmas_encoding(senses=w2v_senses_2, lemmas=w2v_lemmas_2)

#using the two embedding created I try to create a map from semcor to conl
voc_bn_conl= bn_to_conl(w2v_sens=new_bn_embedding+new_conl_embedding, voc_SemCor=voc_lemmas_to_bn_sinsets)

# trasform the bn simbols in Conl simbols
def translate_y(y,voc_bn_conl):
    new_y=[]
    for s in y[0:10000]:
        new_s=[]
        for w in s:
            if w in voc_bn_conl.keys():
                new_s.append(voc_bn_conl[w])
            else:
                new_s.append(w)
        new_y.append(new_s)
    return new_y

y= translate_y(y, voc_bn_conl)


def zip_lpy(lemmas, pos, labels):
    mask=[]
    new_matrix=[]
    for index in range(len(lemmas)):
        mask_sent=[]
        new_sentence=[]
        sentence_l=lemmas[index]
        sentence_p=pos[index]
        sentence_y=labels[index]
        for index_w in range(len(sentence_l)):
            new_word=[sentence_l[index_w],sentence_p[index_w],sentence_y[index_w]]
            new_sentence.append(new_word)
            if sentence_l[index_w]==sentence_y[index_w]:
                mask_sent.append(0)
            else:
                mask_sent.append(1)
        new_matrix.append(new_sentence)
        mask.append(mask_sent)
    return new_matrix,mask

train_bn, mask_train_bn=zip_lpy(l[0:10000],p[0:10000],y)

#TRASFORMO I POS DA PEN TREEBANK A UNIVERSAL
PTB_UNIVERSAL_MAP = tagset_mapping('en-ptb', 'universal')
def pos_to_universal(matrix):
    new_matrix = []
    for sentence in matrix:
        new_sentence = []
        for word in sentence:
            new_word = [word[0], str(PTB_UNIVERSAL_MAP[word[1]]), word[2]]
            new_sentence.append(new_word)
        new_matrix.append(new_sentence)
    return new_matrix

train_matrix=pos_to_universal(train_matrix)
dev_matrix=pos_to_universal(dev_matrix)
test_matrix=pos_to_universal(test_matrix)

#UNISCO CONL E SEMCOR PER AVERE TRAIN SET PIu AMPIO...
train_matrix=train_matrix+train_bn
mask_train=mask_train+mask_train_bn
w2v_lemmas=w2v_lemmas+w2v_lemmas_2

#from now the procesure is the same as SRL:
#build the vocabularies,
#transorm the words,
#build the lemmas embeeding,
#save all the data

def build_vocabolaries(matrix,lemmas_vocab_size):
    voc_lemmas={"UNK":0}
    freq_dict_lemmas={}
    voc_pos={"UNK":0, "_":1}
    voc_pred={"UNK":0}
    ps=2
    #d=2
    pr=1
    #for j in range(len(matrix)):
    for j in range(len(matrix)):
        for i in range(len(matrix[j])):
            word=matrix[j][i]

            if word[0] in freq_dict_lemmas.keys():
                c = freq_dict_lemmas.get(word[0])
                freq_dict_lemmas[word[0]] = c + 1
            else:
                freq_dict_lemmas[word[0]] =1

            if not(word[1] in voc_pos.keys()):
                voc_pos[word[1]]=ps
                ps+=1

            if word[2]=="_":
                if not (word[0] in voc_pred.keys()):
                    voc_pred[word[0]] = pr
                    pr += 1
            else:
                if not (word[2] in voc_pred.keys()):
                    voc_pred[word[2]] = pr
                    pr += 1

    listW_LEMMAS = sorted(freq_dict_lemmas, key=freq_dict_lemmas.get, reverse=True)
    for index in range(1, lemmas_vocab_size):
        if listW_LEMMAS:
            mom = listW_LEMMAS.pop(0)
            voc_lemmas[mom] = index

    return voc_lemmas,voc_pos,voc_pred

voc_lemmas, voc_pos, voc_pred =build_vocabolaries(train_matrix,25000)

dimensions_vocabularies=[len(voc_lemmas),len(voc_pos),len(voc_pred)]

def transform_w2v(matrix):
    input_numb_matrix=[]
    labels_numb_matrix=[]
    for i in range(len(matrix)):
        input_sent = []
        labels_sent=[]
        for j in range(len(matrix[i])):
            word=matrix[i][j]
            #labels_numb_word = []
            input_numb_word =[]

            if word[0] in voc_lemmas.keys():
                input_numb_word.append(voc_lemmas[word[0]])
            else:
                input_numb_word.append(voc_lemmas["UNK"])

            if word[1] in voc_pos.keys():
                input_numb_word.append(voc_pos[word[1]])
            else:
                input_numb_word.append(voc_pos["UNK"])

            if word[2]=="_":
                if word[0] in voc_pred.keys():
                    labels_numb_word=voc_pred[word[0]]
                else:
                    labels_numb_word=voc_pred["UNK"]
            else:
                if word[2] in voc_pred.keys():
                    labels_numb_word=voc_pred[word[2]]
                else:
                    labels_numb_word=voc_pred["UNK"]

            input_sent.append(input_numb_word)
            labels_sent.append(labels_numb_word)
        input_numb_matrix.append(input_sent)
        labels_numb_matrix.append(labels_sent)
    return input_numb_matrix, labels_numb_matrix

input_matrix, labels_matrix= transform_w2v(train_matrix)
dev_input_matrix, dev_labels_matrix=transform_w2v(dev_matrix)
test_matrix,_=transform_w2v(test_matrix)

def build_w2v_embedding(sentences, flag):
    if flag=="lemmas":
        new_sentences=[]
        for s in sentences:
            new_s=[]
            for w in s:
                if w in voc_lemmas.keys():
                    new_s.append(w)
                else:
                    new_s.append("UNK")
            new_sentences.append(new_s)
        model = Word2Vec(new_sentences, size=100, window=5, min_count=1, workers=4)
        listW = sorted(voc_lemmas, key=voc_lemmas.get, reverse=False)
        embedding=[]
        for w in listW:
            print(w,listW.index(w))
            embedding.append(model[w])

    return embedding

lemmas_EMBEDDING= build_w2v_embedding(w2v_lemmas,"lemmas")

flag_save=1
saved_data="conl_dis/"
if flag_save==1:
    print("preprocessing concluso salvataggio dati")

    if os.path.isfile(saved_data+'dim_voc.pckl'):
        os.remove(saved_data+'dim_voc.pckl')
    f = open(saved_data+'dim_voc.pckl', 'wb')
    pickle.dump(dimensions_vocabularies, f)
    f.close()

    if os.path.isfile(saved_data+'X.pckl'):
        os.remove(saved_data+'X.pckl')
    f = open(saved_data+'X.pckl', 'wb')
    pickle.dump(input_matrix, f)
    f.close()
    if os.path.isfile(saved_data+'Y.pckl'):
        os.remove(saved_data+'Y.pckl')
    f = open(saved_data+'Y.pckl', 'wb')
    pickle.dump(labels_matrix, f)
    f.close()
    if os.path.isfile(saved_data+'X_dev.pckl'):
        os.remove(saved_data+'X_dev.pckl')
    f = open(saved_data+'X_dev.pckl', 'wb')
    pickle.dump(dev_input_matrix, f)
    f.close()
    if os.path.isfile(saved_data+'Y_dev.pckl'):
        os.remove(saved_data+'Y_dev.pckl')
    f = open(saved_data+'Y_dev.pckl', 'wb')
    pickle.dump(dev_labels_matrix, f)
    f.close()

    if os.path.isfile(saved_data+'X_test.pckl'):
        os.remove(saved_data+'X_test.pckl')
    f = open(saved_data+'X_test.pckl', 'wb')
    pickle.dump(test_matrix, f)
    f.close()

    if os.path.isfile(saved_data+'lemmas_embedding.pckl'):
        os.remove(saved_data+'lemmas_embedding.pckl')
    f = open(saved_data+'lemmas_embedding.pckl', 'wb')
    pickle.dump(lemmas_EMBEDDING, f)
    f.close()

    if os.path.isfile(saved_data + 'mask_train.pckl'):
        os.remove(saved_data + 'mask_train.pckl')
    f = open(saved_data + 'mask_train.pckl', 'wb')
    pickle.dump(mask_train, f)
    f.close()

    if os.path.isfile(saved_data + 'mask_dev.pckl'):
        os.remove(saved_data + 'mask_dev.pckl')
    f = open(saved_data + 'mask_dev.pckl', 'wb')
    pickle.dump(mask_dev, f)
    f.close()

    if os.path.isfile(saved_data + 'dict_lemmas.pckl'):
        os.remove(saved_data + 'dict_lemmas.pckl')
    f = open(saved_data + 'dict_lemmas.pckl', 'wb')
    pickle.dump(voc_lemmas, f)
    f.close()

    if os.path.isfile(saved_data + 'dict_pred.pckl'):
        os.remove(saved_data + 'dict_pred.pckl')
    f = open(saved_data + 'dict_pred.pckl', 'wb')
    pickle.dump(voc_pred, f)
    f.close()
