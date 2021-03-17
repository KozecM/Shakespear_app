# %tensorflow_version 2.x
import tensorflow as tf 
import tensorflow_datasets.public_api as tfds
import pandas as pd
import numpy as np
import os
import unicodedata
import time
import functools
from six.moves import urllib
from IPython import display as ipythondisplay
from tqdm import tqdm
# download and import the MIT 6.S191 package
import mitdeeplearning as mdl
print(tf.config.list_physical_devices())
assert len(tf.config.list_physical_devices('GPU')) > 0

# load your dataset 
df = tfds.load(name="imdb_reviews", split='train[:5%]')

# df = tfds.load(name='tiny_shakespearee')['train']

df = df.map(lambda x: tf.strings.unicode_split(x['text'], 'UTF-8'))
iter_df = iter(df)
vocabulary = set([])

for review in iter_df:
  temp_vocab = sorted(set(tfds.as_numpy(review)))
  vocabulary.update(temp_vocab)

vocabulary = sorted(vocabulary)
# vocabulary = sorted(set(next(iter(tfds.as_numpy(df)))))

shakespeare = df.map(lambda x: {'cur_char': x[:-1], 'next_char': x[1:]})

char2idx = {u:i for i, u in enumerate(vocabulary)}
idx2char = np.array(vocabulary)

def string_and_vectorized(df):
  vector = []
  temp = []
  shakespeare = []
  current = 0
  length = len(list(df))
  for example in tfds.as_numpy(df): 
    current += 1
    if(current % 10 == 0): print("loop {0}  of  {1}".format(current, length))
    for i in example:
      x = char2idx[i]
      temp = np.append(temp, x)
    shakespeare = example
    vector = np.append(vector, temp)
    temp = []
    vector = vector.astype('int64')
  
  return vector, shakespeare

print("String and Vectorize...")
# shakespeare_vector, shakespeare = string_and_vectorized(df)
def load_npy(filename):
  temp = np.load(filename)
  return(temp)

shakespeare_vector = load_npy('numpy_data/imdb_data.npy')
# np.save('numpy_data/imdb_data.npy', shakespeare_vector)

def get_batch(shakespeare_vector, seq_length, batch_size):
  # the length of the vectorized songs string
  n = shakespeare_vector.shape[0] - 1
  # randomly choose the starting indices for the examples in the training batch
  idx = np.random.choice(n-seq_length, batch_size)
  print("idx:", idx)

  input_batch = [shakespeare_vector[i : i+seq_length] for i in idx]

  output_batch = [shakespeare_vector[i+1 : i+seq_length+1] for i in idx]
  # x_batch, y_batch provide the true inputs and targets for network training
  x_batch = np.reshape(input_batch, [batch_size, seq_length])
  y_batch = np.reshape(output_batch, [batch_size, seq_length])
  return x_batch, y_batch

def LSTM(rnn_units): 
  return tf.keras.layers.LSTM(
    rnn_units, 
    return_sequences=True, 
    recurrent_initializer='glorot_uniform',
    recurrent_activation='sigmoid',
    stateful=True,
  )

### Defining the RNN Model ###

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    LSTM(rnn_units),
    tf.keras.layers.Dense(vocab_size)

  ])

  return model

print("Building Model...")
model = build_model(len(vocabulary), embedding_dim=256, rnn_units=1024, batch_size=32)

model.summary()

print("getting batch")
x, y = get_batch(shakespeare_vector, seq_length=100, batch_size=32)
pred = model(x)

def compute_loss(labels, logits):
  loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True) 
  return loss

example_batch_loss = compute_loss(y, pred) 

print("Prediction shape: ", pred.shape, " # (batch_size, sequence_length, vocab_size)") 
print("scalar_loss:      ", example_batch_loss.numpy().mean())

### Hyperparameter setting and optimization ###

# Optimization parameters:
num_training_iterations = 15000  # Increase this to train longer
batch_size = 5  
seq_length = 400  
learning_rate = 3e-3 

# Model parameters: 
vocab_size = len(vocabulary)
embedding_dim = 256 
rnn_units = 1024  

# Checkpoint location: 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

### Define optimizer and training operation ###

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)


optimizer = tf.keras.optimizers.Adam(learning_rate)

@tf.function
def train_step(x, y): 
  # Use tf.GradientTape()
  with tf.GradientTape() as tape:
  
    y_hat = model(x)
    loss = compute_loss(y, y_hat)

  # Now, compute the gradients 
  grads = tape.gradient(loss, model.trainable_variables)
  
  # Apply the gradients to the optimizer so it can update the model accordingly
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss

##################
# Begin training!#
##################

history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

for iter in tqdm(range(num_training_iterations)):

  # Grab a batch and propagate it through the network
  x_batch, y_batch = get_batch(shakespeare_vector, seq_length, batch_size)
  loss = train_step(x_batch, y_batch)

  # Update the progress bar
  history.append(loss.numpy().mean())
  plotter.plot(history)

  # Update the model with the changed weights!
  if iter % 100 == 0:   
    model.save_weights(checkpoint_prefix)
    
# Save the trained model and the weights
model.save_weights(checkpoint_prefix)


model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

# Restore the model weights for the last checkpoint after training
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()

model.save('models/imdb')

def generate_text(model, start_string, generation_length=2000):
  # Evaluation step (generating ABC text using the learned RNN model)

  input_eval = [char2idx[(bytes(i,encoding='utf8'))] for i in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Here batch size == 1
  model.reset_states()
  tqdm._instances.clear()

  for i in tqdm(range(generation_length)):
    predictions = model(input_eval)
    
    # Remove the batch dimension
    predictions = tf.squeeze(predictions, 0)
    
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    
    # Pass the prediction along with the previous hidden state
    #   as the next inputs to the model
    input_eval = tf.expand_dims([predicted_id], 0)
    text_generated.extend(idx2char[predicted_id])
    
  return (start_string + ''.join(map(chr, text_generated)))


generated_text = generate_text(model, start_string="to be", generation_length=1500)
print()
print(str(generated_text))