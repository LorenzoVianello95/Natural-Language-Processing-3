#SOME AUXILIARES FUNCTIONS USED FOR THE DISAMBIGUATION SYSTEM

#OPEN THE FILES THAT CONTAIN STRING LIKE senseval2.d000.s002.t000  bn:00030861n AND BUILDS DICTIONARIES
def build_idtowsense(file):
    d = {}
    with open(file, 'r') as f:
        for line in f:
            (key, val) = line.split(" ")
            d[key] = val.replace('\n', '')
    return d


stoplist = set(["wf"])#,"lemma","pos"
stopwords = set([w.rstrip('\r\n') for w in open("StopwordsM.txt")])

#from xml file extract the sentences, divided by type:lemmas, pos, tokens, labels..
#in the sentences I choose of don't put punctuation and the numbers are replaced by "NUM"
#lemmas: list of list dove ogni sottolista rappresenta una sentence e
#          al suo interno ci sono tutti i lemmas di quella sentence in ordine
#poses: list of list dove ogni sottolista rappresenta una sentence e
#          al suo interno ci sono tutti i poses di quella sentence in ordine
#           inoltre se il lemma corrispondente e' nome, adjettivo , o verbo
#           allora sostituisco direttamente con il suo corrispondente valore all'interno di d_idtws
#tokens: list of list dove ogni sottolista rappresenta una sentence e
#          al suo interno ci sono tutti i tokens di quella sentence in ordine
#maxlen= lunghezza massima delle sentence, mi serve per costruire i placeholder in maniera adeguata
def convert(xml_file,d_idtws):
    lemmas=[]
    poses = []
    tokens = []
    wsd=[]
    wsd_onlyId=[]
    voc_l_s={}
    maxlen=0
    with open(xml_file, "r") as f:  # notice the "rb" mode
        lemmas_s = []
        poses_s= []
        tokens_s = []
        wsd_s = []
        wsd_onlyId_s=[]
        for line in f.readlines():
            if True:#len(lemmas)< 50000:
                line = line.replace('>', ' ').replace('<', ' ').replace('"', ' ').replace('=', ' ').replace('/', ' ')
                split = line.strip().split()
                split = [word for word in split if (word not in stoplist) and (len(word) > 0)]
                if len(split)>0:
                    if split[0]=="sentence":
                        if len(lemmas_s)>0:
                            lemmas.append(lemmas_s)
                            poses.append(poses_s)
                            tokens.append(tokens_s)
                            wsd.append(wsd_s)
                            wsd_onlyId.append(wsd_onlyId_s)
                            if len(lemmas_s)>maxlen:
                                #print(lemmas_s)
                                maxlen=len(lemmas_s)
                            lemmas_s = []
                            poses_s = []
                            tokens_s = []
                            wsd_s=[]
                            wsd_onlyId_s=[]
                    elif split[0] == "text" or split[0] == "corpus" or split[0] == "?xml" or split[0] == "det":
                        pass
                    elif len(split)>3 and split[1] == "id":
                        #print(split)
                        poses_s.append(split[6])
                        wsd_s.append(d_idtws[split[2]])
                        wsd_onlyId_s.append(d_idtws[split[2]])
                        lemmas_s.append(split[4])
                        voc_l_s[d_idtws[split[2]]]=split[4]
                        tokens_s.append(split[7])
                    elif len(split)==5:
                        if split[3]== "NUM":
                            poses_s.append("NUM")
                            lemmas_s.append("NUM")
                            wsd_s.append("NUM")
                            tokens_s.append("NUM")
                            wsd_onlyId_s.append(0)
                        elif split[3]== ".":
                            pass
                        elif split[1] in ['&#178;','&apos;em', '&apos;ll', '&amp;amp;', '&apos;', '&apos;d', '&apos;s', '**f', '**f-value', '-gt']:
                            pass
                        else:
                            if split[1] not in stopwords:
                                poses_s.append(split[3])
                                lemmas_s.append(split[1])
                                wsd_s.append(split[1])
                                tokens_s.append(split[4])
                                wsd_onlyId_s.append(0)
    return lemmas,poses,tokens,maxlen,wsd,wsd_onlyId, voc_l_s


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

#CREATE SENTENCES THAT HAVE INFORMATION ABOUT LEMMAS AND PREDS
def unify_sens_lemmas_encoding(senses,lemmas):
    new_encoding=[]
    for index_sent in range(len(senses)):
        new_sent = []
        sent = senses[index_sent]
        sent_lemma = lemmas[index_sent]
        for index_w in range(len(sent)):
            if sent[index_w] == sent_lemma[index_w]:
                new_sent.append(sent[index_w])
            else:
                new_sent.append(sent[index_w])
                new_sent.append(sent_lemma[index_w])
        new_encoding.append(new_sent)
    return new_encoding


from gensim.models import Word2Vec


#MY ATTEMPT TO LINK SEMCOR AND CONL USING THE W2V EMBEEDING
def bn_to_conl(w2v_sens,voc_SemCor):
    model_mom = Word2Vec(w2v_sens, size=100, window=10, min_count=5, workers=4)
    word_vectors = model_mom.wv
    voc_bn_conl={}
    c=0
    for word in voc_SemCor.keys():
        word1 = voc_SemCor[word]
        if word in word_vectors.vocab and word1 in word_vectors.vocab:
            vector=model_mom[word]
            vector1 = model_mom[word1]
            nw = model_mom.most_similar([vector,vector1], [], 50)
            for el in nw:
                if not el[0].startswith("bn"):
                    if el[0].find(".0")>0:
                        if el[0].find(word1)>=0:
                            voc_bn_conl[word]=el[0]
                            c+=1
                            #print(word,word1,el)
                            break
            if not word in voc_bn_conl.keys():
                voc_bn_conl[word]=word1
                #print(word,word1)
        else:
            voc_bn_conl[word] = word1
    #print(c)
    return voc_bn_conl