import tensorflow as tf 
import tensorflow_datasets.public_api as tfds
from tqdm import tqdm
import numpy as np
import os

df = tfds.load(name='tiny_shakespeare')['train']
df = tfds.load(name="imdb_reviews", split='train[:5%]')
df = df.map(lambda x: tf.strings.unicode_split(x['text'], 'UTF-8'))
iter_df = iter(df)
vocabulary = set([])

for review in iter_df:
  temp_vocab = sorted(set(tfds.as_numpy(review)))
  vocabulary.update(temp_vocab)


vocabulary = sorted(vocabulary)
char2idx = {u:i for i, u in enumerate(vocabulary)}
idx2char = np.array(vocabulary)
print(idx2char)

model = tf.keras.models.load_model('models/imdb')

def generate_text(model, start_string, generation_length=2000):
  # Evaluation step (generating ABC text using the learned RNN model)

  input_eval = [char2idx[(bytes(i,encoding='utf8'))] for i in start_string]
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
      text_generated.extend(idx2char[predicted_id])
    
  return (start_string + ''.join(map(chr, text_generated)))


generated_text = generate_text(model, start_string="I am", generation_length=1500)
print()
print(str(generated_text))