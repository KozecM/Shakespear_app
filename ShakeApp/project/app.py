from flask import Flask, render_template, request, flash, redirect, url_for, session
import tensorflow as tf 
from tqdm import tqdm
import tensorflow_datasets.public_api as tfds
import numpy as np

import os

# config
SECRET_KEY = "temp"
PLAY = ""

app= Flask(__name__)

app.config.from_object(__name__)

class g_vars():
  df = None 
  vocabulary = None 
  char2idx = None 
  idx2char = None

def set_data(type):
  if(type == 'shakespeare'):
    g_vars.df = tfds.load(name='tiny_shakespeare')['train']
  else:
    g_vars.df = tfds.load(name='imdb_reviews', split='train[:5%]')
  set_vocab()

def set_vocab():
  g_vars.df = g_vars.df.map(lambda x: tf.strings.unicode_split(x['text'], 'UTF-8'))

  iter_df = iter(g_vars.df)
  g_vars.vocabulary = set([])

  for review in iter_df:
    temp_vocab = sorted(set(tfds.as_numpy(review)))
    g_vars.vocabulary.update(temp_vocab)

  g_vars.vocabulary = sorted(g_vars.vocabulary)
  g_vars.char2idx = {u:i for i, u in enumerate(g_vars.vocabulary)}
  g_vars.idx2char = np.array(g_vars.vocabulary)


def generate_text(model, start_string, generation_length=2000):
  # Evaluation step (generating ABC text using the learned RNN model)

  input_eval = [g_vars.char2idx[(bytes(i,encoding='utf8'))] for i in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Here batch size == 1
  model.reset_states()
  if hasattr(tqdm, '_instances'): tqdm._instances.clear()

  for i in tqdm(range(generation_length)):
      predictions = model(input_eval)
      
      # Remove the batch dimension
      predictions = tf.squeeze(predictions, 0)
      
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      
      # Pass the prediction along with the previous hidden state
      #   as the next inputs to the model
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.extend(g_vars.idx2char[predicted_id])
    
  return (start_string + ''.join(map(chr, text_generated)))

@app.route("/")
def index():
  script = session.get('script', "")
  return render_template('index.html', script = script)

@app.route('/write', methods=['POST'])
def write_shakespear():
  # use the model to write shakespear
  prompt = request.form['prompt']
  length = request.form['length']
  model_type = request.form['model']

  set_data(model_type)
  model = tf.keras.models.load_model('project/models/' + model_type)

  session['script'] = generate_text(model, prompt, int(length))

  flash('Here is your play!') if model_type == 'shakespeare' else flash('Here is your movie review!') 
  return redirect(url_for('index') )

if __name__ == "__main__":
  app.run()