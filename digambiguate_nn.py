#RETE NEURALE PER DISAMBIGUATE SIA CONL CHE SemCor

import tensorflow as tf
import pickle
import numpy as np
np.set_printoptions(threshold=np.inf)
import tqdm
import sys
import math
import os

#Split sentences too long in shorter sentences
def split_sentences_wsd(bx, by, msk, max_dim):
    new_bx = []
    new_by = []
    new_msk=[]
    for index in range(len(bx)):
        sent=bx[index]
        lab=by[index]
        m=msk[index]
        if (len(sent)==len(lab)):
            if len(sent) < max_dim:
                new_bx.append(sent)
                new_by.append(lab)
                new_msk.append(m)
            else:
                div=(len(sent)//max_dim)+1
                for i in range(div):
                    d = [sent[i%len(sent)] for i in range(i*((len(sent)//div)+1),(i+1)*((len(sent)//div)+1))]
                    f=  [lab[i%len(sent)] for i in range(i*((len(sent)//div)+1),(i+1)*((len(sent)//div)+1))]
                    g= [m[i%len(sent)] for i in range(i*((len(sent)//div)+1),(i+1)*((len(sent)//div)+1))]
                    new_bx.append(d)
                    new_by.append(f)
                    new_msk.append(g)
        else:
            print("LUNGHEZZE DIVERSE")
    return new_bx, new_by, new_msk

#padding
def create_padding_x(matrix, max_length): #pero non ha senso fare split sent...
    #split phase:
    new_matrix=[]
    pad_word = [0, 0]
    for s in matrix:
        #print(s)
        if len(s)<max_length:
            while len(s)<max_length:
                s.append(pad_word)
            new_matrix.append(s)
        else:
            new_matrix.append(s)
    return new_matrix

#fill the sentences too short using a zero padding
def create_padding_y(listoflist, max_length):
    cb_lol= []
    for s in listoflist:
        data_s =list(s)
        while len(data_s)<max_length:
            data_s.append(0)
        cb_lol.append(data_s)

    return cb_lol

saved_data="conl_dis/"
print(saved_data)
#RESTORE ALL THE PREPROCESSED DATA
f = open(saved_data+'X.pckl', 'rb')
X = pickle.load(f)
f.close()
f = open(saved_data+'Y.pckl', 'rb')
Y = pickle.load(f)
f.close()
f = open(saved_data+'dim_voc.pckl', 'rb')
[len_voc_lemmas,len_voc_pos,len_voc_pred] = pickle.load(f)
print(len_voc_lemmas,len_voc_pos,len_voc_pred)
f.close()
#dev:
f = open(saved_data+'X_dev.pckl', 'rb')
X_dev = pickle.load(f)
f.close()
f = open(saved_data+'Y_dev.pckl', 'rb')
Y_dev = pickle.load(f)
f.close()

f = open(saved_data+'lemmas_embedding.pckl', 'rb')
lemmas_embedding = pickle.load(f)
f.close()

f = open(saved_data+'mask_train.pckl', 'rb')
mask_train = pickle.load(f)
f.close()

f = open(saved_data+'mask_dev.pckl', 'rb')
mask_dev = pickle.load(f)
f.close()

f = open(saved_data+'dict_lemmas.pckl', 'rb')
dict_lemmas = pickle.load(f)
f.close()

f = open(saved_data+'dict_pred.pckl', 'rb')
dict_pred = pickle.load(f)
f.close()

print("recupero dei dati concluso")

#NN HYPERPARAMETERS
hidden_size= 300

max_sentencence_dimension=30
NUM_STEPS=len(X)
batch_size=15
lr=0.001
ntags=len_voc_pred
encoding_size= 2#len_voc_lemmas+len_voc_pos
depth_list= [len_voc_lemmas, len_voc_pos]

print("number tags",ntags)


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
    with tf.name_scope('mask_pred'):
        # shape = (batch, sentence)
        mask_pred = tf.placeholder(tf.int32, shape=[None, None], name="mask_pred")


    L = tf.Variable(np.array(lemmas_embedding), dtype=tf.float32, trainable=False)
    tensor_lemmas = tf.nn.embedding_lookup(L, word_ids[:,:, 0])
    print(tensor_lemmas.shape)

    tensor_pos = tf.one_hot(word_ids[:,:, 1], depth_list[1], axis=2)
    print(tensor_pos.shape)

    sentences_encoded = tf.concat([tensor_lemmas, tensor_pos], axis=2)

    with tf.name_scope("LSTM_layer"):
        cell_fw = tf.contrib.rnn.LSTMCell(hidden_size)
        cell_bw = tf.contrib.rnn.LSTMCell(hidden_size)

        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
            cell_bw, sentences_encoded, sequence_length=sequence_lengths,
            dtype=tf.float32)

        context_rep = tf.concat([output_fw, output_bw], axis=-1)#lstmoutput2


        W = tf.get_variable("W", shape=[2*hidden_size, ntags],
                        dtype=tf.float32)

        b = tf.get_variable("b", shape=[ntags], dtype=tf.float32,
                        initializer=tf.zeros_initializer())

        ntime_steps = tf.shape(context_rep)[1]
        context_rep_flat = tf.reshape(context_rep, [-1, 2*hidden_size])
        pred = tf.matmul(context_rep_flat, W) + b
        scores = tf.reshape(pred, [-1, ntime_steps, ntags])

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

    with tf.name_scope('mask_2'):
        zero = tf.constant(0, dtype=tf.int32)
        mask_2 = tf.not_equal(mask_pred, zero)
        l_p=tf.boolean_mask(labels_pred, mask_2)
        l=tf.boolean_mask(labels, mask_2)

    with tf.name_scope('WSD_accuracy'):
        correct_pred_wsd = tf.equal(l_p, l)
        accuracy_wsd= tf.reduce_mean(tf.cast(correct_pred_wsd, tf.float32))
        tf.summary.scalar('WSD_accuracy', accuracy_wsd)


    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()

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

        #mask used to identify where are the predicates
        train_m= [mask_train[i % NUM_SENTECES] for i in range(step, step + batch_size)]

        batch_x, batch_y, train_m = split_sentences_wsd(batch_x, batch_y, train_m, max_sentencence_dimension)

        len_sent = [len(sx) for sx in batch_x]
        max_len_sent = max_sentencence_dimension

        batch_x=create_padding_x(batch_x,max_len_sent)
        batch_y = create_padding_y(batch_y, max_len_sent)
        train_m= create_padding_y(train_m, max_len_sent)

        sess.run(train_op, feed_dict={word_ids: batch_x,
                                      sequence_lengths:len_sent,
                                      labels: batch_y,
                                      mask_pred:train_m
                                      })
        #display the values and saves on tensorboard
        if step % display_step == 0 :

            summary_str = sess.run(summary, feed_dict={word_ids: batch_x,
                                      sequence_lengths:len_sent,
                                      labels: batch_y,mask_pred:train_m})
            writer.add_summary(summary_str, step)
            writer.flush()

            l_correct = sess.run(l, feed_dict={word_ids: batch_x,
                                                   sequence_lengths: len_sent,
                                                   labels: batch_y, mask_pred: train_m})
            l_predict = sess.run(l_p, feed_dict={word_ids: batch_x,
                                               sequence_lengths: len_sent,
                                               labels: batch_y, mask_pred: train_m})

            ac = sess.run(accuracy_wsd, feed_dict={word_ids: batch_x,
                                                             sequence_lengths: len_sent,
                                                             labels: batch_y, mask_pred: train_m})

            print("Step " + str(step)  + ", ACcuracy= "+str(ac) )
            #print(pred)
            #print(l)
            #print(batch_y)

    print("Optimization Finished!")

    #FUNCTION THAT CALCULATE THE ACCURACY FOR AMBIGOUS WORDS PREDICTION IN DEV
    def testWSD(Xd,Yd):
        accc=[]
        prediction=[]
        bar = tqdm.tqdm(range(0,len(Xd),batch_size))
        for index in bar:
            #print(len(Xd),len(Yd),len(mask_dev))
            batch_x = [Xd[i % len(Xd)] for i in range(index, index +batch_size)]
            batch_y = [Yd[i % len(Xd)] for i in range(index, index+batch_size)]
            #len_sent = [len(Xd[i % len(Xd)]) for i in range(index, index+batch_size)]
            dev_m = [mask_dev[i % len(Xd)] for i in range(index,index + batch_size)]

            batch_x, batch_y, dev_m = split_sentences_wsd(batch_x, batch_y, dev_m, max_sentencence_dimension)
            len_sent = [len(sx) for sx in batch_x]
            max_len_sent = max_sentencence_dimension
            batch_x = create_padding_x(batch_x, max_len_sent)
            batch_y = create_padding_y(batch_y, max_len_sent)
            dev_m = create_padding_y(dev_m, max_len_sent)

            pred = sess.run(labels_pred_m, feed_dict={word_ids: batch_x,
                                                      sequence_lengths: len_sent,
                                                      labels: batch_y,
                                                      mask_pred:dev_m})

            res = dict((v, k) for k, v in dict_pred.iteritems())

            for word in pred:
                prediction.append(res[word])
            a= sess.run(accuracy_wsd, feed_dict={word_ids: batch_x,
                                                      sequence_lengths: len_sent,
                                                      labels: batch_y,
                                                      mask_pred:dev_m})
            accc.append(a)

        print("batchsize: ", batch_size, " Num steps: ", NUM_STEPS, " hidden size: ", hidden_size )
        accc=np.array(accc)
        print("Precision= " + str(np.average(accc)) )

        f.close()

    testWSD(X_dev,Y_dev)

   # __________ test phase_____________________
    f = open(saved_data + 'X_test.pckl', 'rb')
    X_test = pickle.load(f)
    f.close()
    reversed_lemmas_voc = dict((v, k) for k, v in dict_lemmas.items())
    reversed_preds_voc = dict((v, k) for k, v in dict_pred.items())

    with open(saved_data+'test_dis_labelling.txt', 'w') as f:

        for index_b in range(len(X_test)):
            sentenc=X_test[index_b]
            batch_x = [X_test[index_b]]
            len_sent = [len(X_test[index_b])]
            batch_y=[[0 for _ in range(len(X_test[index_b]))]]
            mask_test=[[0 for _ in range(len(X_test[index_b]))]]

            pred = sess.run(labels_pred_m, feed_dict={word_ids: batch_x,
                                                      sequence_lengths: len_sent,
                                                      labels: batch_y,
                                                      mask_pred: mask_test})


            f.write("\n")
            f.write(str(index_b) + "len_sentence: " + str(len(sentenc)))
            f.write("\n")
            for index in range(len(X_test[index_b])):
                f.write(reversed_lemmas_voc[sentenc[index][0]] + '\t'+ reversed_preds_voc[pred[index]])
                f.write("\n")
            f.write("\n")
    f.close()
