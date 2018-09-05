import tensorflow as tf
import numpy as np
import pandas as pd
import tflearn
from sklearn.model_selection import train_test_split
import data_helper
from keras.utils import to_categorical
from keras.preprocessing.sequence import skipgrams, pad_sequences
from random import shuffle
import pickle
import tqdm
import argparse
import os

#save dataframe with pickle
def save_df_pickle(df, name) :
    with open(name, 'wb') as f :
        pickle.dump(df, f)

#load dataframe with pickle
def load_df_pickle(name) :
    with open(name, 'rb') as f :
        df = pickle.load(f)
    return df


def model():
    # Batch size list of integer sequences
    x = tf.placeholder(tf.int32, shape = [None, sequence_length], name="x")
    tf.add_to_collection('x', x)
    # One hot labels for sentiment classification
    y = tf.placeholder(tf.int32, shape = [None, num_classes], name="y")
    # Cast our label to float32
    y = tf.cast(y, tf.float32)
    tf.add_to_collection('y', y)
    # Instantiate our embedding matrix
    Embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                            name="word_embedding")
    tf.add_to_collection('Embedding', Embedding)
    # Lookup embeddings
    embed_lookup = tf.nn.embedding_lookup(Embedding, x, name='embed_lookup')
    tf.add_to_collection('embed_lookup', embed_lookup)
    # Create LSTM/GRU Cell
    cell = tf.contrib.rnn.GRUCell(hidden_units)
    #cell = tf.contrib.rnn.BasicLSTMCell(hidden_units)

    # Extract the batch size - this allows for variable batch size
    current_batch_size = tf.shape(x)[0]

    # Create Initial State of Zeros for memory cell
    initial_state = cell.zero_state(current_batch_size, dtype=tf.float32)

    # Wrap our lstm cell in a dropout wrapper
    cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.85)
    tf.add_to_collection('cell', cell)
    value, _ = tf.nn.dynamic_rnn(cell,
                                 embed_lookup,
                                 initial_state=initial_state,
                                 dtype=tf.float32)

    #Instantiate weights1
    weight1 = tf.Variable(tf.random_normal([hidden_units, int((hidden_units + num_classes)/2)]), name='weight1')
    #Instantiate biases1
    bias1 = tf.Variable(tf.constant(0.1, shape=[int((hidden_units + num_classes)/2)]), name='bias1')
    #Instantiate weights2
    weight2 = tf.Variable(tf.random_normal([int((hidden_units + num_classes)/2), num_classes]), name='weight2')
    #Instantiate biases1
    bias2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='bias2')

    tf.add_to_collection('weight1', weight1)
    tf.add_to_collection('bias1', bias1)
    tf.add_to_collection('weight2', weight2)
    tf.add_to_collection('bias2', bias2)

    value = tf.transpose(value, [1,0,2], name='value')
    tf.add_to_collection('value', value)
    #Extract last output
    last = tf.gather(value, int(value.get_shape()[0])-1, name='last')
    tf.add_to_collection('last', last)

    hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(last, weight1), bias1, name='hidden_layer'))
    prediction = tf.add(tf.matmul(hidden_layer, weight2), bias2, name='prediction')
    tf.add_to_collection('prediction', prediction)

    correct_prediction = tf.equal(tf.argmax(tf.nn.sigmoid(prediction), axis=1), tf.argmax(y, axis=1), name='correct_prediction')
    tf.add_to_collection('correct_prediction', correct_prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.add_to_collection('accuracy', accuracy)
    choice = tf.argmax(tf.nn.sigmoid(prediction), axis=1, name='choice')
    tf.add_to_collection('choice', choice)
    # Calculate the loss given prediction and labels
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction,
                                                                     labels = y), name='loss')
    tf.add_to_collection('loss', loss)

    # Declare our optimizer, in this case RMS Prop
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, name='optimizer').minimize(loss)
    tf.add_to_collection('optimizer', optimizer)
    return optimizer, loss, x, y, accuracy, prediction, correct_prediction, choice


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Select data')

    parser.add_argument('-train', dest='train', default=None,
                       help='training file')
    parser.add_argument('-test', dest='test', default=None,
                       help='testing file')

    args = parser.parse_args()

    is_train_valid = False
    is_test_valid = False

    if args.train != None :
        if os.path.isfile("selected_data/" + args.train) :
            print("valid training data")
            is_train_valid = True
        else :
            print('no file detected')
    else :
        print('please set a training dataframe with -train')

    if args.test != None :
        if os.path.isfile("selected_data/" + args.test) :
            print("valid testing data")
            is_test_valid = True
        else :
            print('no file detected')
    else :
        print('please set a testing dataframe with -test')


    if is_train_valid and is_test_valid :
        df = load_df_pickle('selected_data/' + args.train)

        text = df['LongDescription'].values
        labels = df['ID-Famile'].values

        # vocab size
        vocab_size = 1000
        # Size of our embedding matrix
        embedding_size = 100
        # Number of samples for NCE Loss
        num_samples = 64
        # Learning Rate
        learning_rate = 0.001
        # Number of hidden units
        hidden_units = 100
        # Number of classes
        num_classes = 20
        # Number of words in each of our sequences
        sequence_length = 50
        # Batch size
        batch_size = 32
        #
        num_epochs = 30

        data, word_to_idx, idx_to_word, T = data_helper.tokenize_and_process(text, vocab_size = vocab_size)
        data = pad_sequences(data, maxlen=sequence_length)
        labels = to_categorical(labels, num_classes = num_classes)
        save_df_pickle(T, 'selected_data/T') #this is the training tokenizer, need to reuse it for proper testing processing


        x_train, testx, y_train, testy = train_test_split(data, labels, test_size=0.0)

        optimizer, loss, x, y, accuracy, prediction, correct_prediction, choice = model()

        num_batches = len(x_train) // batch_size

        with tf.Session() as sess :
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)

            # create a saver and FileWriter
            saver = tf.train.Saver()
            writer = tf.summary.FileWriter("log/", graph = sess.graph)

            for epoch in tqdm.tqdm(range(num_epochs)) :
                print("---Epoch {} out of {}---".format(epoch+1, num_epochs))
                if epoch > 0 :
                    data = list(zip(x_train, y_train))
                    shuffle(data)
                    x_train, y_train = zip(*data)

                for i in range(num_batches) :
                    if i != num_batches-1 :
                        x_batch = x_train[i*batch_size:i*batch_size+batch_size]
                        y_batch = y_train[i*batch_size:i*batch_size+batch_size]
                    else :
                        x_batch = x_train[i*batch_size:]
                        y_batch = y_train[i*batch_size:]
                    #print(y_batch)
                    _, l, a = sess.run([optimizer, loss, accuracy], feed_dict={x: x_batch, y: y_batch})

                    if i % 100 == 0 :
                        print("step {} of {}, loss : {}, accuracy : {}".format(i, num_batches, l, a))
                        saver.save(sess, "model/model")
                        writer.flush()
                        writer.close()

            #l, a = sess.run([loss, accuracy], feed_dict={x: testx, y: testy})
            #print('test set acc', a)


            # TEST SUR LES DONNÉES
            dftest = load_df_pickle('selected_data/' + args.test)
            text = dftest['LongDescription'].values
            labels = dftest['ID-Famile'].values

            data = data_helper.tokenize_and_test(text, T)
            data = pad_sequences(data, maxlen=sequence_length)
            labels = to_categorical(labels, num_classes = num_classes)

            x_train, testx, y_train, testy = train_test_split(data, labels, test_size=0.0)


            c, l, a = sess.run([choice, loss, accuracy], feed_dict={x: x_train, y: y_train})
            for i in range(len(y_train)) :
                print(y_train[i], c[i])
            print('acc', a,'loss', l)

    else :
        print("can't train and test")
