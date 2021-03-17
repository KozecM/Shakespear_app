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

def test_write_shakespeare(client):
  rv =client.post(
    "/write",
    data=dict(prompt="Test", length="20", model = 'shakespeare'),
    follow_redirects = True
  )

  assert b"Test" in rv.data

def test_write_imdb(client):
  rv = client.post(
    "/write",
    data=dict(prompt="Test", length="20", model = 'imdb'),
    follow_redirects=True
  )
  assert b"Test" in rv.data

def test_data_loaded():
  set_data('shakespeare')

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

def test_shakespeare_model_exists():
  assert os.path.exists("project/models/shakespeare/saved_model.pb")

def test_imdb_model_exists():
  assert os.path.exists("project/models/imdb/saved_model.pb")

def test_shakespeare_message():
  rv = client.post(
    "/write",
    data=dict(prompt="Test", length="5", model = 'shakespeare'),
    follow_redirects=True
  )
  assert b"Here is your play!" in rv.data

def test_shakespeare_message():
  rv = client.post(
    "/write",
    data=dict(prompt="Test", length="5", model = 'imdb'),
    follow_redirects=True
  )
  assert b"Here is your movie review!" in rv.data