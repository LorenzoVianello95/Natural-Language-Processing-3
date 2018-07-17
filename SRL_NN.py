import tensorflow as tf
import pickle
import numpy as np
np.set_printoptions(threshold=np.inf)
import tqdm

#auxliary function that tell in what position pred with flag=1 is
def find_pred(sentence):
    i=len(sentence)//2
    for index in range(len(sentence)):
        word=sentence[index]
        if word[4]==1:
            i=index
    return i

#function that split the sentences, the new sentence are the words around the pred
#with flag=1, the dimension of new sentence is 2* window_size
#this function return also if doing this passage we exclude sono flag different from zero,
#so return the number of them as True Negative
def split_sentences(bx,by,ws):
    new_bx=[]
    new_by=[]
    fn = 0
    for index in range(len(bx)):
        sent=bx[index]
        lab=by[index]
        if len(sent)<ws*2+1:
            new_bx.append(sent)
            new_by.append(lab)
        else:
            p=find_pred(sent)
            new_sent=[sent[i%len(sent)] for i in range(p-ws,p+ws)]
            new_lab = [lab[i % len(sent)] for i in range(p - ws, p + ws)]
            new_bx.append(new_sent)
            new_by.append(new_lab)
            for lab_index in range(len(lab)):
                if lab_index<p-ws or lab_index>=p+ws:
                    if not lab[lab_index]==0:
                        fn+=1
    return new_bx,new_by,fn

#create padding for input words adding the false words [0, 0, 0, 1, 0]
def create_padding_x(matrix, max_length):
    #split phase:
    new_matrix=[]
    pad_word = [0, 0, 0, 1, 0]
    for s in matrix:
        #print(s)
        if len(s)>max_length:
            d = [s[i] for i in range(0, round(len(s)/2))]
            e = [s[i] for i in range(round(len(s) / 2), len(s))]
            new_matrix.append(d)
            new_matrix.append(e)
        elif len(s)<max_length:
            new_pad_word=pad_word
            while len(new_pad_word)<len(s[0]):
                new_pad_word.append(0)
            while len(s)<max_length:
                s.append(new_pad_word)
            new_matrix.append(s)
        else:
            new_matrix.append(s)
    return new_matrix

#fill the labels sentences  too short using a zero padding
def create_padding_y(listoflist, max_length):
    cb_lol= []
    for s in listoflist:
        data_s =list(s)
        while len(data_s)<max_length:
            data_s.append(0)
        cb_lol.append(data_s)

    return cb_lol

#RESTORE ALL THE PREPROCESSED DATA
f = open('X.pckl', 'rb')
X = pickle.load(f)
f.close()
f = open('Y.pckl', 'rb')
Y = pickle.load(f)
f.close()
f = open('1hotencoding.pckl', 'rb')
[len_voc_lemmas,len_voc_pos,len_voc_deprel,len_voc_pred,len_voc_apreds] = pickle.load(f)
print(len_voc_lemmas,len_voc_pos,len_voc_deprel,len_voc_pred,len_voc_apreds)
f.close()
#dev:
f = open('X_dev.pckl', 'rb')
X_dev = pickle.load(f)
f.close()
f = open('Y_dev.pckl', 'rb')
Y_dev = pickle.load(f)
f.close()

f = open('lemmas_embedding.pckl', 'rb')
lemmas_embedding = pickle.load(f)
f.close()
f = open('senses_embedding.pckl', 'rb')
senses_embedding = pickle.load(f)
f.close()

print(len(lemmas_embedding))
print(len(senses_embedding))
print("recupero dei dati concluso")

#NN HYPERPARAMETERS
hidden_size= 300
window_size= 20
max_dim_sentences=window_size*2
NUM_STEPS=2*len(X)
batch_size=20
lr=0.001
ntags=len_voc_apreds
encoding_size= 5#len_voc_lemmas+len_voc_pos+len_voc_deprel+len_voc_pred+2
depth_list= [len_voc_lemmas, len_voc_pos, len_voc_deprel, len_voc_pred, 2]


#NN CONSTRUCTION:
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('inputs_words'):
        # shape = (batch size, max length of sentence in batch,5 encoding number for word)
        word_ids = tf.placeholder(tf.int32, shape=[None, None,encoding_size],name="words_ids")
    with tf.name_scope('lenght_sentences'):
        # shape = (batch size)
        sequence_lengths = tf.placeholder(tf.int32, shape=[None])
    with tf.name_scope('output_labels'):
        # shape = (batch, sentence)
        labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

    #ENCODING LEMMAS USING THE W2V EMBEEDING
    L = tf.Variable(np.array(lemmas_embedding), dtype=tf.float32, trainable=False)
    tensor_lemmas = tf.nn.embedding_lookup(L, word_ids[:,:, 0])

    #ENCODING POS USING 1 HOT ENCODING
    tensor_pos = tf.one_hot(word_ids[:,:, 1], depth_list[1], axis=2)
    #ENCODING DEPREL USING 1 HOT
    tensor_deprel = tf.one_hot(word_ids[:,:, 2], depth_list[2], axis=2)

    # ENCODING PREDICATES USING THE W2V EMBEEDING
    S = tf.Variable(np.array(senses_embedding), dtype=tf.float32, trainable=False)
    tensor_pred = tf.nn.embedding_lookup(S, word_ids[:,:, 3])

    #FLAG CAN BE ONLY 0 OR 1 SO 1 HOT ENCODING WITH 2 ELEMENTS
    tensor_flag = tf.one_hot(word_ids[:,:, 4], depth_list[4], axis=2)

    #FINAL SENTENCE IS THE CONCATENATION OF ALL THESE ENCODING
    sentences_encoded = tf.concat([tensor_lemmas, tensor_pos, tensor_deprel, tensor_pred, tensor_flag], axis=2)

    with tf.name_scope("LSTM_layer"):
        cell_fw = tf.contrib.rnn.LSTMCell(hidden_size)
        cell_bw = tf.contrib.rnn.LSTMCell(hidden_size)

        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
            cell_bw, sentences_encoded, sequence_length=sequence_lengths,
            dtype=tf.float32)

        context_rep = tf.concat([output_fw, output_bw], axis=-1)


        W = tf.get_variable("W", shape=[2*hidden_size, ntags],
                        dtype=tf.float32)

        b = tf.get_variable("b", shape=[ntags], dtype=tf.float32,
                        initializer=tf.zeros_initializer())

        ntime_steps = tf.shape(context_rep)[1]
        context_rep_flat = tf.reshape(context_rep, [-1, 2*hidden_size])
        pred = tf.matmul(context_rep_flat, W) + b
        scores = tf.reshape(pred, [-1, ntime_steps, ntags])#lstmoutput2

    # # ___________ATTENTIVE LAYER_____________________
    #
    # input_shape = context_rep.get_shape().as_list()
    #
    # with tf.name_scope("Attentive_layer"):
    #     w_a = tf.Variable(tf.random_uniform(shape=(input_shape[-1], 1), minval=-1, maxval=1))
    #     b_a = tf.zeros(shape=[ntime_steps])
    #     b_a = tf.Variable(b_a, validate_shape=False)
    #
    #     u=tf.tensordot(context_rep,w_a,axes=1)
    #     u=tf.squeeze(u,-1)
    #     u=u+b_a
    #     u=tf.tanh(u)
    #     a=tf.exp(u)
    #
    #     mask = tf.sequence_mask(sequence_lengths,ntime_steps)
    #     if mask is not None:
    #         mask=tf.cast(mask,tf.float32)
    #         a=mask*a
    #
    #     a/= tf.cast(tf.reduce_sum(a,axis=1,keepdims=True)+ tf.keras.backend.epsilon() , tf.float32) #K.epsilon()
    #
    #     focused_feats=tf.tensordot(a,u)
    #     t=[focused_feats,a]
    #
    #     weighted_input=context_rep*tf.expand_dims(a,-1)
    #     output=tf.reduce_sum(weighted_input,axis=1)
    #
    #     att_output=tf.tile(output,[1,ntime_steps])
    #     att_output=tf.reshape(att_output,(-1,ntime_steps,2*hidden_size))
    #
    # #_________CONCATENATION LSTM OUTPUT E ATT LAYER________
    #
    # with tf.name_scope("Concatenation"):
    #     att_output= tf.concat([att_output,scores],-1)
    #     print("att_o",att_output.get_shape().as_list())

    #_________SOFTMAX LAYER________________________________

    #scores=att_output
    with tf.name_scope("softmax_layer"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=labels)

    with tf.name_scope('mask_1'):
        mask = tf.sequence_mask(sequence_lengths)
        losses = tf.boolean_mask(losses, mask)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(losses)
        tf.summary.scalar('loss', loss)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss)

    with tf.name_scope('prediction'):
        labels_pred = tf.cast(tf.argmax(scores, axis=-1), tf.int32)
        labels_pred_m=tf.boolean_mask(labels_pred, mask)
        labels_m=tf.boolean_mask(labels, mask)

    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()

#training phase
print("session")
display_step=100
with tf.Session(graph=graph) as sess:
    writer = tf.summary.FileWriter("/home/lollo/Desktop/NLP/log", sess.graph)
    # Run the initializer
    NUM_SENTECES=len(X)
    sess.run(init)
    bar = tqdm.tqdm(range(0, NUM_STEPS, batch_size))
    for step in bar:
        batch_x = [X[i % NUM_SENTECES] for i in range(step, step + batch_size)]
        batch_y = [Y[i%NUM_SENTECES] for i in range(step,step+batch_size)]

        #split the sentences too longs
        batch_x,batch_y, FN =split_sentences(batch_x,batch_y,window_size)

        #array that contain the lengths of the sentences
        len_sent = [len(sx) for sx in batch_x]
        max_len_sent = max_dim_sentences

        #pad the sentences all to the same length
        batch_x=create_padding_x(batch_x,max_len_sent)
        batch_y = create_padding_y(batch_y, max_len_sent)

        sess.run(train_op, feed_dict={word_ids: batch_x,
                                      sequence_lengths:len_sent,
                                      labels: batch_y})

        #display the values and saves on tensorboard
        if step % display_step == 0 :

            summary_str = sess.run(summary, feed_dict={word_ids: batch_x,
                                      sequence_lengths:len_sent,
                                      labels: batch_y})
            writer.add_summary(summary_str, step)
            writer.flush()


            pred = sess.run(labels_pred_m, feed_dict={word_ids: batch_x,
                                                              sequence_lengths:len_sent,
                                                              labels: batch_y})
            l = sess.run(labels_m, feed_dict={word_ids: batch_x,
                                                      sequence_lengths: len_sent,
                                                      labels: batch_y})
            fp=0
            tp=0
            fn=FN
            for index in range(len(l)):
                if pred[index]==0:
                    if not l[index]==0:
                        fn+=1
                else:
                    if pred[index]==l[index]:
                        tp+=1
                    else:
                        fp+=1
            P=0
            R=0
            F1=0
            if not tp+fp==0:
                P=float(tp)/(tp+fp)
            if not tp + fn == 0:
                R=float(tp)/(tp+fn)
            if not P+R == 0:
                F1 = 2 * (P * R) / (P + R)
            print("Step " + str(step)  + ", Precision= "+str(P)+", Recall= "+str(R)+", F1= "+str(F1) )
            #print(pred)
            #print(l)
            #print(batch_y)

    print("Optimization Finished!")

    #devition phase, calculate precision, recall, f1
    def testWSD(Xd,Yd):
        fp = 0
        tp = 0
        fn = 0
        confusion_matrix = np.zeros((len_voc_apreds, len_voc_apreds))
        bar = tqdm.tqdm(range(0,len(Xd),batch_size))
        for index in bar:
            batch_x = [Xd[i % len(Xd)] for i in range(index, index +batch_size)]
            batch_y = [Yd[i % len(Xd)] for i in range(index, index+batch_size)]

            batch_x, batch_y, FN = split_sentences(batch_x, batch_y, window_size)
            len_sent = [len(sx) for sx in batch_x]
            max_len_sent = max_dim_sentences

            fn+=FN

            batch_x = create_padding_x(batch_x, max_len_sent)
            batch_y = create_padding_y(batch_y, max_len_sent)

            pred = sess.run(labels_pred_m, feed_dict={word_ids: batch_x,
                                                      sequence_lengths: len_sent,
                                                      labels: batch_y})
            l = sess.run(labels_m, feed_dict={word_ids: batch_x,
                                              sequence_lengths: len_sent,
                                              labels: batch_y})
            for index in range(len(l)):
                if pred[index]==0:
                    if not l[index]==0:
                        fn+=1
                else:
                    if pred[index]==l[index]:
                        tp+=1
                    else:
                        fp+=1


            for index in range(len(l)):
                confusion_matrix[pred[index], l[index]]+=1

        return tp,fp,fn, confusion_matrix
    flag = 0
    if (flag == 0):
        tp,fp,fn,c_m = testWSD(X_dev, Y_dev)
        P = 0
        R = 0
        F1 = 0
        if not tp + fp == 0:
            P = float(tp) / (tp + fp)
        if not tp + fn == 0:
            R = float(tp) / (tp + fn)
        if not P + R == 0:
            F1 = 2 * (P * R) / (P + R)

        print("batchsize: ", batch_size, " Num steps: ", NUM_STEPS, " hidden size: ", hidden_size )
        print("Precision= " + str(P) + ", Recall= " + str(R) + ", F1= " + str(F1))

        f = open('voc_apreds.pckl', 'rb')
        voc_apreds = pickle.load(f)
        f.close()
        reversed_apreds_voc = dict((v, k) for k, v in voc_apreds.items())

        print([reversed_apreds_voc[i] for i in range(len(reversed_apreds_voc))])
        for p in c_m.tolist():
            print p


    elif flag==2:
        #_____________________TESTING PHASE___________________

        f = open('voc_lemmas.pckl', 'rb')
        voc_lemmas = pickle.load(f)
        f.close()
        f = open('voc_apreds.pckl', 'rb')
        voc_apreds = pickle.load(f)
        f.close()
        f = open('testset.pckl', 'rb')
        testset = pickle.load(f)
        f.close()

        print("starting testing:")

        reversed_lemmas_voc = dict((v, k) for k, v in voc_lemmas.items())
        reversed_apreds_voc = dict((v, k) for k, v in voc_apreds.items())

        X_test = testset

        set_of_batches = []  # un batch per ogni sentence
        for sent in X_test:
            batch = []
            for i in range(len(sent)):
                word = sent[i]
                if word[4] > 0:
                    new_sent = []
                    for index_before in range(i):
                        if sent[index_before][4] == 0:
                            new_sent.append(sent[index_before])
                        else:
                            new_word = [sent[index_before][0], sent[index_before][1], sent[index_before][2],
                                        sent[index_before][3], 0]
                            new_sent.append(new_word)
                    new_word = [word[0], word[1], word[2], word[3], 1]
                    new_sent.append(new_word)
                    for index_after in range(i + 1, len(sent)):
                        if sent[index_after][4] == 0:
                            new_sent.append(sent[index_after])
                        else:
                            new_word = [sent[index_after][0], sent[index_after][1], sent[index_after][2],
                                        sent[index_after][3], 0]
                            new_sent.append(new_word)
                    batch.append(new_sent)
            set_of_batches.append(batch)

    
        with open('test_labelling.txt', 'w') as f:

            for index_b in range(len(set_of_batches)):
                test_sentenc=X_test[index_b]
                b=set_of_batches[index_b]
                if len(b)>0:
                    batch_x= b
                    len_sent=[len(s) for s in b]

                    max_len_sent=max(len_sent)

                    l_p = sess.run(labels_pred_m, feed_dict={word_ids: batch_x,
                                                       sequence_lengths: len_sent})


                    reversed_lemmas_voc=dict((v,k) for k,v in voc_lemmas.items())
                    reversed_apreds_voc = dict((v, k) for k, v in voc_apreds.items())
                    f.write("\n")
                    f.write(str(index_b)+"len_sentence: "+str(max_len_sent)+" number of pred= "+str(len(b))+"\n")
                    f.write("\n")
                    for index in range(max_len_sent):
                        f.write(reversed_lemmas_voc[test_sentenc[index][0]]+ '\t')
                        for el in [l_p[j] for j in range(index,len(l_p),max_len_sent)]:
                            f.write(reversed_apreds_voc[el] + '\t')
                        f.write("\n")
                    f.write("\n")

                else:
                    for index in range(len(test_sentenc)):
                        f.write(reversed_lemmas_voc[test_sentenc[index][0]]+ '\n')
                    f.write("\n")
        f.close()

print("sessione conclusa")
