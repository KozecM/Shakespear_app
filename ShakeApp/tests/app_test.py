import pytest
import os
from pathlib import Path
from project.app import app
from project.app import generate_text
from project.app import set_data
from project.app import g_vars
import tensorflow as tf 


@pytest.fixture
def client():
  BASE_DIR = Path(__file__).resolve().parent.parent
  app.config["TESTING"] = True

  yield app.test_client()


def test_index():
  tester = app.test_client()
  response = tester.get("/", content_type="html/text")

  assert response.status_code == 200

def test_shakespeare_model():
  assert Path("project/models/shakespeare/saved_model.pb").is_file()

def test_imdb_model():
  assert Path("project/models/imdb/saved_model.pb").isfile()

def test_write_shakespeare_page(client):
  rv =client.post(
    "/write",
    data=dict(prompt="Hello", length="20", model = 'shakespeare'),
    follow_redirects = True
  )

  assert b"Hello" in rv.data

def test_data_loaded():
  set_data('shakespeare')

  print (type(g_vars.idx2char))
  assert g_vars.df != None
  assert g_vars.vocabulary != None

def test_shakespeare_model():
  set_data('shakespeare')
  model = tf.keras.models.load_model('project/models/shakespeare')
  result = generate_text(model, "Test", 20)

  assert len(result) == 24
  assert "Test" in result  

def test_imdb_model():
  set_data('imdb')
  model = tf.keras.models.load_model('project/models/imdb')
  result = generate_text(model, "Test", 20)

  assert len(result) == 24
  assert "Test" in result 