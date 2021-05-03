# Summarizing Text
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
print('TensorFlow Version: {}'.format(tf.__version__))

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


"""## Insepcting the Data"""

reviews = pd.read_csv("drive/Colab Notebooks/Menu/Data/Reviews.csv")

reviews.shape

reviews.head()

# Check for any nulls values
reviews.isnull().sum()

# Remove null values and unneeded features
reviews = reviews.dropna()
reviews = reviews.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator',
                        'Score','Time'], 1)
reviews = reviews.reset_index(drop=True)

reviews.head()

# Inspecting some of the reviews
for i in range(5):
    print("Review #",i+1)
    print(reviews.Summary[i])
    print(reviews.Text[i])
    print()

"""## Preparing the Data"""

contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

def clean_text(text, remove_stopwords = True):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    
    # Convert words to lower case
    text = text.lower()
    
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text

import nltk
nltk.download('stopwords')
  
# Clean the summaries and texts
clean_summaries = []
for summary in reviews.Summary:
    clean_summaries.append(clean_text(summary, remove_stopwords=False))
print("Summaries are complete.")

clean_texts = []
for text in reviews.Text:
    clean_texts.append(clean_text(text))
print("Texts are complete.")

# Inspect the cleaned summaries and texts to ensure they have been cleaned well
for i in range(5):
    print("Clean Review #",i+1)
    print(clean_summaries[i])
    print(clean_texts[i])
    print()

def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1

# Find the number of times each word was used and the size of the vocabulary
word_counts = {}

count_words(word_counts, clean_summaries)
count_words(word_counts, clean_texts)
            
print("Size of Vocabulary:", len(word_counts))

# Load Conceptnet Numberbatch's (CN) embeddings, similar to GloVe, but probably better 
# (https://github.com/commonsense/conceptnet-numberbatch)
embeddings_index = {}
with open('drive/Colab Notebooks/Menu/Data/numberbatch-en-17.02.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings:', len(embeddings_index))

# Find the number of words that are missing from CN, and are used more than our threshold.
missing_words = 0
threshold = 20

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1
            
missing_ratio = round(missing_words/len(word_counts),4)*100
            
print("Number of words missing from CN:", missing_words)
print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

"""I use a threshold of 20, so that words not in CN can be added to our word_embedding_matrix, but they need to be common enough in the reviews so that the model can understand their meaning."""

# Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe

#dictionary to convert words to integers
vocab_to_int = {} 

value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

print("Total number of unique words:", len(word_counts))
print("Number of words we will use:", len(vocab_to_int))
print("Percent of words we will use: {}%".format(usage_ratio))

# Need to use 300 for embedding dimensions to match CN's vectors.
embedding_dim = 300
nb_words = len(vocab_to_int)

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        # If word not in CN, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(vocab_to_int)
print(len(word_embedding_matrix))

def convert_to_ints(text, word_count, unk_count, eos=False):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count

# Apply convert_to_ints to clean_summaries and clean_texts
word_count = 0
unk_count = 0

int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count)
int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, eos=True)

unk_percent = round(unk_count/word_count,4)*100

print("Total number of words in headlines:", word_count)
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))

def create_lengths(text):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])

lengths_summaries = create_lengths(int_summaries)
lengths_texts = create_lengths(int_texts)

print("Summaries:")
print(lengths_summaries.describe())
print()
print("Texts:")
print(lengths_texts.describe())

# Inspect the length of texts
print(np.percentile(lengths_texts.counts, 90))
print(np.percentile(lengths_texts.counts, 95))
print(np.percentile(lengths_texts.counts, 99))

# Inspect the length of summaries
print(np.percentile(lengths_summaries.counts, 90))
print(np.percentile(lengths_summaries.counts, 95))
print(np.percentile(lengths_summaries.counts, 99))

def unk_counter(sentence):
    '''Counts the number of time UNK appears in a sentence.'''
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
    return unk_count

# takes a long time  , this is normal

# Sort the summaries and texts by the length of the texts, shortest to longest
# Limit the length of summaries and texts based on the min and max ranges.
# Remove reviews that include too many UNKs

sorted_summaries = []
sorted_texts = []
max_text_length = 84
max_summary_length = 13
min_length = 2
unk_text_limit = 1
unk_summary_limit = 0

for length in range(min(lengths_texts.counts), max_text_length): 
    for count, words in enumerate(int_summaries):
        if (len(int_summaries[count]) >= min_length and
            len(int_summaries[count]) <= max_summary_length and
            len(int_texts[count]) >= min_length and
            unk_counter(int_summaries[count]) <= unk_summary_limit and
            unk_counter(int_texts[count]) <= unk_text_limit and
            length == len(int_texts[count])
           ):
            sorted_summaries.append(int_summaries[count])
            sorted_texts.append(int_texts[count])
        
# Compare lengths to ensure they match
print(len(sorted_summaries))
print(len(sorted_texts))

"""## Building the Model"""

def model_inputs():
    '''Create palceholders for inputs to the model'''
    
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
    max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
    text_length = tf.placeholder(tf.int32, (None,), name='text_length')

    return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length

def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input

def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    '''Create the encoding layer'''
    
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
                                                    input_keep_prob = keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
                                                    input_keep_prob = keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                    cell_bw, 
                                                                    rnn_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)
    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output,2)
    
    return enc_output, enc_state

def training_decoding_layer(dec_embed_input, summary_length, dec_cell, initial_state, output_layer, 
                            vocab_size, max_summary_length):
    '''Create the training logits'''
    
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=summary_length,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       training_helper,
                                                       initial_state,
                                                       output_layer) 

    training_logits, _ , _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                           output_time_major=False,
                                                           impute_finished=True,
                                                           maximum_iterations=max_summary_length)
    return training_decoder

def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_summary_length, batch_size):
    '''Create the inference logits'''
    
    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
    
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)
                
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        initial_state,
                                                        output_layer)
                
    inference_logits, _ , _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                            output_time_major=False,
                                                            impute_finished=True,
                                                            maximum_iterations=max_summary_length)
    
    return inference_decoder

def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length, 
                   max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):
    '''Create the decoding cell and attention for the training and inference decoding layers'''
    
    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                     input_keep_prob = keep_prob)
    
    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                  enc_output,
                                                  text_length,
                                                  normalize=False,
                                                  name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                          attn_mech,
                                                          rnn_size)
            
    #initial_state = tf.contrib.seq2seq.AttentionWrapperState(enc_state[0],
    #                                                                _zero_state_tensors(rnn_size, 
    #                                                                                    batch_size, 
    #                                                                                    tf.float32)) 
    initial_state = dec_cell.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=enc_state[0])

    with tf.variable_scope("decode"):
        training_decoder = training_decoding_layer(dec_embed_input, 
                                                  summary_length, 
                                                  dec_cell, 
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size, 
                                                  max_summary_length)
        
        training_logits,_ ,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                  output_time_major=False,
                                  impute_finished=True,
                                  maximum_iterations=max_summary_length)
    with tf.variable_scope("decode", reuse=True):
        inference_decoder = inference_decoding_layer(embeddings,  
                                                    vocab_to_int['<GO>'], 
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell, 
                                                    initial_state, 
                                                    output_layer,
                                                    max_summary_length,
                                                    batch_size)
        
        inference_logits,_ ,_ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                  output_time_major=False,
                                  impute_finished=True,
                                  maximum_iterations=max_summary_length)

    return training_logits, inference_logits

def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length, 
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):
    '''Use the previous functions to create the training and inference logits'''
    
    # Use Numberbatch's embeddings and the newly created ones as our embeddings
    embeddings = word_embedding_matrix
    
    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)
    
    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
    
    training_logits, inference_logits  = decoding_layer(dec_embed_input, 
                                                        embeddings,
                                                        enc_output,
                                                        enc_state, 
                                                        vocab_size, 
                                                        text_length, 
                                                        summary_length, 
                                                        max_summary_length,
                                                        rnn_size, 
                                                        vocab_to_int, 
                                                        keep_prob, 
                                                        batch_size,
                                                        num_layers)
    
    return training_logits, inference_logits

def pad_sentence_batch(sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]



def get_batches(summaries, texts, batch_size):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts)//batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch))
        
        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))
        
        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))
        
        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths

# Set the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 256
num_layers = 2
learning_rate = 0.005
keep_probability = 0.75

# Build the graph
train_graph = tf.Graph()
# Set the graph to default to ensure that it is ready for training
with train_graph.as_default():
    
    # Load the model inputs    
    input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()

    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                      targets, 
                                                      keep_prob,   
                                                      text_length,
                                                      summary_length,
                                                      max_summary_length,
                                                      len(vocab_to_int)+1,
                                                      rnn_size, 
                                                      num_layers, 
                                                      vocab_to_int,
                                                      batch_size)
    
    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
    
    # Create the weights for sequence_loss
    masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
print("Graph is built.")

"""## Training the Model

Since I am training this model on my MacBook Pro, it would take me days if I used the whole dataset. For this reason, I am only going to use a subset of the data, so that I can train it over night. Normally I use [FloydHub's](https://www.floydhub.com/) services for my GPU needs, but it would take quite a bit of time to upload the dataset and ConceptNet Numberbatch, so I'm not going to bother with that for this project.

I chose not use use the start of the subset because I didn't want to make it too easy for my model. The texts that I am using are closer to the median lengths; I thought this would be more fair.
"""

# Subset the data for training
start = 200000
end = start + 300000
sorted_summaries_short = sorted_summaries[start:end]
sorted_texts_short = sorted_texts[start:end]
print("The shortest text length:", len(sorted_texts_short[0]))
print("The longest text length:",len(sorted_texts_short[-1]))

# Train the Model
learning_rate_decay = 0.95
min_learning_rate = 0.0005
display_step = 20 # Check training loss after every 20 batches
stop_early = 0 
stop = 6 #3 # If the update loss does not decrease in 3 consecutive update checks, stop training
per_epoch = 3 # Make 3 update checks per epoch
update_check = (len(sorted_texts_short)//batch_size//per_epoch)-1

update_loss = 0 
batch_loss = 0
summary_update_loss = [] # Record the update losses for saving improvements in the model

  
tf.reset_default_graph()
checkpoint = "drive/Colab Notebooks/Menu/Model 300/best_model.ckpt"  #300k sentence
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    

    for epoch_i in range(1, epochs+1):
        update_loss = 0
        batch_loss = 0
        for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                get_batches(sorted_summaries_short, sorted_texts_short, batch_size)):
            start_time = time.time()
            _, loss = sess.run(
                [train_op, cost],
                {input_data: texts_batch,
                 targets: summaries_batch,
                 lr: learning_rate,
                 summary_length: summaries_lengths,
                 text_length: texts_lengths,
                 keep_prob: keep_probability})

            batch_loss += loss
            update_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time

            if batch_i % display_step == 0 and batch_i > 0:
                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                      .format(epoch_i,
                              epochs, 
                              batch_i, 
                              len(sorted_texts_short) // batch_size, 
                              batch_loss / display_step, 
                              batch_time*display_step))
                batch_loss = 0
                
                #saver = tf.train.Saver() 
                #saver.save(sess, checkpoint)
                
            if batch_i % update_check == 0 and batch_i > 0:
                print("Average loss for this update:", round(update_loss/update_check,3))
                summary_update_loss.append(update_loss)
                
              
                  
                # If the update loss is at a new minimum, save the model
                if update_loss <= min(summary_update_loss):
                    print('New Record!') 
                    stop_early = 0
                    saver = tf.train.Saver() 
                    saver.save(sess, checkpoint)

                else:
                    print("No Improvement.")
                    stop_early += 1
                    if stop_early == stop:
                        break
                update_loss = 0
            
                    
        # Reduce learning rate, but not below its minimum value
        learning_rate *= learning_rate_decay
        if learning_rate < min_learning_rate:
            learning_rate = min_learning_rate
        
        if stop_early == stop:
            print("Stopping Training.")
            break

checkpoint = "drive/Colab Notebooks/Menu/Model Backup/best_model.ckpt" 

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)
    names = []
    [names.append(n.name) for n in loaded_graph.as_graph_def().node]
names

"""## Making Our Own Summaries

To see the quality of the summaries that this model can generate, you can either create your own review, or use a review from the dataset. You can set the length of the summary to a fixed value, or use a random value like I have here.
"""

def text_to_seq(text):
    '''Prepare the text for the model'''
    
    text = clean_text(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]


random = np.random.randint(0,len(clean_texts))
input_sentence = clean_texts[random]
text = text_to_seq(clean_texts[random])

checkpoint = "drive/Colab Notebooks/Menu/Model 300/best_model.ckpt"

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    text_length = loaded_graph.get_tensor_by_name('text_length:0')
    summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    
    #Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(logits, {input_data: [text]*batch_size, 
                                      summary_length: [np.random.randint(5,8)], 
                                      text_length: [len(text)]*batch_size,
                                      keep_prob: 1.0})[0] 

# Remove the padding from the tweet
pad = vocab_to_int["<PAD>"] 

print('Original Text:', reviews.Text[random])
print('Original summary:', reviews.Summary[random])#clean_summaries[random]

print('\nText')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([int_to_vocab[i] for i in text])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))

"""Examples of reviews and summaries:
- Review(1): The coffee tasted great and was at such a good price! I highly recommend this to everyone!
- Summary(1): great coffee


- Review(2): This is the worst cheese that I have ever bought! I will never buy it again and I hope you won't either!
- Summary(2): omg gross gross


- Review(3): love individual oatmeal cups found years ago sam quit selling sound big lots quit selling found target expensive buy individually trilled get entire case time go anywhere need water microwave spoon know quaker flavor packets
- Summary(3): love it

**My Examples  (model 150k)**
- Review(1): I am not a big fan of cereal but it is nice to have it at the house.  Both my husband and my kid likes this.  I personally prefer protein shake for breakfast.
- Original summary(1): Family likes it
- My Summary(1): very nice breakfast cereal
---
- Original Text(2): Good value for a lot of treats! my dog absolutely loves these and it's awesome that they sell them on amazon for half the price of what they would sell it for at a pet store! Will continue buying from here as long as they're available :)
- Original summary(2): good value!
- My Summary(2): awesome value
---
- Original Text(3): My baby seems to prefer this formula mixture over the other bigger brands. If you buy it "Subscribe and Save" it's a lot cheaper, which is nice. I breastfed for 9 months until my baby stopped gaining weight so we had to start giving her formula. She's been growing steadily ever since!
- Original summary(3): Seems healthy, and my baby prefers it
- My Summary(3): great formula for baby
---
- Original Text(4): These are so good, especially when they are still hot.  I love the way they still sizzle and pop while eating.  Sometimes I sprinkle some hot sauce on mine straight from the microwave or a little parmesan cheese, depending on my mood.  However I eat them, they are really good.  Remember to shake them well for a minute or two before nuking them.  The bags do not puff up as much as microwave popcorn, so don't expect them to.  Only cook for the suggested time on the bag.
- Original summary(4): Yummmy!
- My Summary(4) : good stuff
---
- Original Text(5): IF YOU WANT A HIGH ENERGY, AND GREAT TASTING (AND I MEAN THE VERY BEST)ENERGY DRINKS, THIS IS THE ONE FOR YOU! BEVNET.COM SAYS IT TASTES LIKE MOUNTAIN DEW, AND IT DOES! VOTED BEST TASTING! HAS JIFFY INGREDIENT THAT MAKES THIS DIFFERENT CALLED D-RIBOSE, AND IT IS A SIMPLE SUGAR THAT IS GREAT FOR ATHLETES! NO BAD THINGS HERE. I HAD A BAD BATCH ONCE, BUT IT WAS A FLUKE.
- Original summary(5): THESE ARE THE BEST!
- My Summary(5) : the best energy drink
---
- Original Text(6): What I really appreciate about these bars is that it really takes time to eat and enjoy them. Highly recommended!
- Original summary(6): Great product!
- My Summary(6) : foolproof boys bars

---

- Original Text(7): When I was growing up, we ate a fair amount of Spam.  Every so often, I revisit those days, by purchasing some Spam and trying to enjoy it. And I think of the old Monty Python segment focusing on Spam. . . .<br /><br />It's not so easy for me to enjoy! Plain Spam, even cooked, doesn't do it for me.  I start out with good intentions of eating it as part of a meal and normally throw quite a bit of it out. So, this time around, I tried something different--a fried Spam sandwich.<br /><br />I cut up and fried some shallots, slathered a couple pieces of bread with Dijon mustard and catsup, and melted some American cheese over the Spam.  And it worked! This time, I successfully fixed something that didn't feature the Spam taste.  The result was an enjoyable sandwich.<br /><br />So, instead of throwing out all the Spam (as has happened in the past), I'm going to use my imagination this time in preparing this for a meal. In the past, this would have rated 1 to 2 stars; this time around, I'll give it three.
- Original summary(7): Back to the future with Spam
- Response Words(7): not the best
---
- Original Text(8): My dog absolutely loves these treats, and the price on these was nearly 1/2 the price I pay at the pet or grocery store!!! Plus I don't have to worry about running out for quite a while, or shall I say my dog does not have to worry!!
- Original summary(8): Great deal!
- Response Words(8): dog loves these
---
- Original Text(9): These Ziggies are great to stuff the Kong.  My dog hates his crate, so it keeps him very busy trying to get it out.<br />Sometimes it takes him hours.
- Original summary(9): Great item
- Response Words(9): dog loves these
---
- Original Text(10): Got some matcha in Japan from the airport and it was surprisingly tasty.  Still, you are told not to steep green tea for more than a couple minutes or else the bitter flavor comes out.  So, with matcha it's just gonna happen, since you can't take out the tea bag.<br /><br />That said, the Japan matcha was more granular than a "powder", more like sand really, and did not produce much bitterness even after ten minutes in hot water.  This Taiwan tea by comparison is a true powder.  It is difficult to dissolve, clumping a quite a bit, and produces a flavor that within a few minutes in hot water becomes bitter, then later even more bitter.<br /><br />UPDATE: It turns out you use a lot less of it than you do of the Japanese matcha I had.  The powder, since it is so fine, goes much further.  To help you with this, they have a spoon in the box to right-size the servings.  I was using a much larger teaspoon instead previously.  Using the smaller amount, it works great for hot tea and does not become bitter.
- Original summary(10): Price is OK, flavor a bit bitter
- Response Words(10): not as good as i hoped
---
- Original Tex(11)t: Oh, this is some good cat food!<br />Have you ever opened a can of Friskies and gagged at the awful smell? How can your cat eat that?<br />This cat food smells pretty good, and it simply looks and smells like good food. My kittens made so much noise when I opened that can! They gobbled it right up, and I have to admit that it smelled pretty good.
- Original summary(11): Exceptional Cat Food!
- Response Words(11): smells good

---
- Original Text(12): Hmm, at first, I thought this was another one of those beverages competing with Red Bull.  However, as soon as I had my first sip, I knew I was in for something healthier.<br /><br />Hmmm...healthy vs. energy boost?  That is the question of the day.<br /><br />The Switch contains 100% fruit juice and none of the other stuff.  It's just carbonated juice.<br /><br />However, I didn't really quite catch that 'black cherry' flavor.  It tasted more like apple juice than anything else.  Nonetheless, it was still good!
- Original summary(12): carbonated juice
- Response Words(12): not like cherry
---
- Original Text: First let me say that these beans taste just fine, however, there are way too many damaged beans in each tin and we were not expecting that.
- Original summary: A little disappointed
- Response Words: darker beans
---
- Original Text: The reviews were what sold me on trying this product & the ones I read raved about the taste, so I thought someone should tell the truth.  I'm an avid coffee drinker, so I just wasn't ready for what I found.  It was horrible!!!
- Original summary: Horrible taste!!!!!!!
-  Response Words: worst coffee i have ever tried so
---



### Model 300k
- Original Text: My little Chihuahua and Jack Russell mix just loves these! She gobbles them up! They are a perfect size for her small mouth. According to her, they are very tasty.
- Original summary: My Dog Loves Them!
- Response Words: perfect size for our retriever
--
- Original Text: I bought this to serve after I picked some blueberries.  This dessert was a huge hit!  Not to sweet, cooks in 1/2 hour and is good hot or cold.  Also the box is so cute I gave it to my kids teachers as a gift!  They loved it!  Just like homemade.
- Original summary: So easy, so good and all natural
- Response Words: great hot snack

## Bleu Test
"""

from nltk.translate.bleu_score import sentence_bleu
reference = [['So', 'easy', 'so', 'good','and','all','natural']]
candidate = ['great', 'hot', 'snack']
score = sentence_bleu(reference, candidate)
print(score)


from sumeval.metrics.rouge import RougeCalculator

refrence_summary = "So easy, so good and all natural"
model_summary = "great hot snack"

rouge = RougeCalculator(stopwords=True, lang="en")

rouge_1 = rouge.rouge_n(
            summary=model_summary,
            references=refrence_summary,
            n=1)

rouge_2 = rouge.rouge_n(
            summary=model_summary,
            references=[refrence_summary],
            n=2)

rouge_l = rouge.rouge_l(
            summary=model_summary,
            references=[refrence_summary])


rouge_be = rouge.rouge_be(
            summary=model_summary,
            references=[refrence_summary])

print("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}, ROUGE-BE: {}".format(
    rouge_1, rouge_2, rouge_l, rouge_be
).replace(", ", "\n"))

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()

sentence = ("""My little Chihuahua and Jack Russell mix just loves these! She 
gobbles them up! They are a perfect
size for her small mouth. According
to her, they are very tasty.""","")
model = ("My Dog Loves Them!","")
summary = ("Chihuahua mix perfect","")

tfidf_matrix_sentence = tfidf_vectorizer.fit_transform(sentence) 
tfidf_matrix_model = tfidf_vectorizer.transform(model) 
tfidf_matrix_summary = tfidf_vectorizer.transform(summary) 


cosine = cosine_similarity(tfidf_matrix_sentence, tfidf_matrix_model)
print(cosine)

cosine = cosine_similarity(tfidf_matrix_sentence, tfidf_matrix_summary)
print(cosine)

"""## Summary

I hope that you found this project to be rather interesting and informative. One of my main recommendations for working with this dataset and model is either use a GPU, a subset of the dataset, or plenty of time to train your model. As you might be able to expect, the model will not be able to make good predictions just by seeing many reviews, it needs so see the reviews many times to be able to understand the relationship between words and between descriptions & summaries. 

In short, I'm pleased with how well this model performs. After creating numerous reviews and checking those from the dataset, I can happily say that most of the generated summaries are appropriate, some of them are great, and some of them make mistakes. I'll try to improve this model and if it gets better, I'll update my GitHub.

Thanks for reading!
"""

