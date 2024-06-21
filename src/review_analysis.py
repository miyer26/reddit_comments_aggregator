import yaml
import os
import requests
import json
from dotenv import load_dotenv

from transformers import pipeline, DistilBertTokenizerFast, DistilBertForTokenClassification
import torch
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def fetch_data(config):
    load_dotenv()
    config['anthropic_api_key'] = os.getenv('ANTHROPIC_API_KEY')
    config['reddit_credentials']['client_id'] = os.getenv('REDDIT_CLIENT_ID')
    config['reddit_credentials']['client_secret'] = os.getenv('REDDIT_CLIENT_SECRET')
    config['reddit_credentials']['username'] = os.getenv('REDDIT_USERNAME')
    config['reddit_credentials']['password'] = os.getenv('REDDIT_PASSWORD')
    config['reddit_credentials']['user_agent'] = os.getenv('REDDIT_USER_AGENT')
    return config

def load_sentiment_model(model_name: str):
  tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
  model = DistilBertForTokenClassification.from_pretrained(model_name)
  classifier = pipeline('sentiment-analysis', model=model_name, tokenizer=tokenizer)

  return classifier

def load_anthropic_model(model_name: str = "claude-3-opus-20240229", api_key=None, **kwargs):
    """
    Load the specified model with optional additional parameters.

    Parameters:
    - model_name (str): The name of the model to load. Defaults to "claude-3-opus-20240229".
    - api_query: The query to use for the API call.
    - kwargs: Additional keyword arguments to pass to the load_sentiment_model function.

    Returns:
    - model: The loaded model.
    """
    model = ChatAnthropic(model_name, api_key, **kwargs)
    return model

def augment_prompt(llm, query):
  human_message = f"You are a reddit user. \
  Given a specific user question, produce 10 unique variants of this question that have the same meaning. \
  Do not repeat the same quesiton. \n \
  Here is the question: {query}. \
  Post the output as a list seperated by commas, without numbering"

  messages = [HumanMessage(content=human_message)]

  response = llm.invoke(messages)
  return response.content.split(',')

def get_auth_token(client_id, client_secret, username, password, user_agent):
  auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
  data = {
    'grant_type': 'password',
    'username': username,
    'password': password
  }
  headers = {'User-Agent': user_agent}
  res = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=headers)
  token = res.json()['access_token']
  return headers, token


search_query_parameters = {
                          'sort': 'relevance',
                           'limit': 5,
                           'restrict_sr': True}

subreddit = 'whatcarshouldIbuy'

comment_parameters = {'depth': 1, 'limit': 5, 'sort': 'top'}

def get_reddit_posts_and_comments(headers, token, search_query_parameters, classifier, queries, comment_parameters,subreddit=None):

  # Use the access token to make a request
  headers = {**headers, **{'Authorization': f'bearer {token}'}}

  if subreddit:
    search = f'https://oauth.reddit.com/r/{subreddit}/search'
    search_query_parameters['restrict_sr'] = True
  else:
    search = f'https:/oauth.reddit.com/search'
    search_query_parameters['restrict_sr'] = False


  posts = []
  for query in queries:
    search_query_parameters['q'] = query
    response = requests.get(search, headers=headers, params=search_query_parameters)
    posts.extend(response.json()['data']['children'])

  reddit_data = {}
  seen_ids = set()
  for post in posts:
      post_data = post['data']
      post_id = post_data['id']

      if post_id in seen_ids: #remove redundant posts
          continue
      seen_ids.add(post_id)

      post_title = post_data['title']
      post_text = post_data['selftext']
      post_url = f"https://reddit.com{post_data['permalink']}"

      # Fetch comments for each post
      subreddit = post_data['subreddit']
      comments_url = f'https://oauth.reddit.com/r/{subreddit}/comments/{post_id}'
      comments_response = requests.get(comments_url, headers=headers, params=comment_parameters)

      try:
          comments = comments_response.json()[1]['data']['children']
      except (IndexError, KeyError):
          comments = []

      comment_list = []
      for comment in comments:
          comment_data = comment['data']
          if 'body' in comment_data:
              comment_text = comment_data['body']
              url = f"https://reddit.com{comment_data['permalink']}"
              result = classifier(comment_text[:512])
              sentiment = result[0]['label']
              confidence = result[0]['score']
              comment_list.append({
                  'comment': comment_text,
                  'sentiment': sentiment,
                  'confidence': confidence,
                  'url': url
              })

      reddit_data[post_id] = {
          'title': post_title,
          'text': post_text,
          'url': post_url,
          'comments': comment_list
      }
  return reddit_data

from langchain.schema import (
    HumanMessage,
    SystemMessage
)

def produce_response_from_reddit_data(llm, system_message, reference_data):

  messages = [
      SystemMessage(content=system_message),
      HumanMessage(content=reference_data)
  ]

  response = llm.invoke(messages)

  return response.content

def generate_review(sentiment_model, llm_model, llm_temperature, anthropic_api_key, reddit_search_query, reddit_credentials, search_query_parameters, comment_parameters, system_message, subreddit=None):
  sentiment_model = load_sentiment_model("distilbert-base-uncased-finetuned-sst-2-english")
  llm = ChatAnthropic(model=llm_model, api_key=anthropic_api_key, temperature=llm_temperature)
  search_query_aug = augment_prompt(llm, reddit_search_query)
  headers, token = get_auth_token(reddit_credentials['client_id'], reddit_credentials['client_secret'], reddit_credentials['username'], reddit_credentials['password'], reddit_credentials['user_agent'])
  reddit_posts = get_reddit_posts_and_comments(headers, token, search_query_parameters, sentiment_model, search_query_aug,comment_parameters, subreddit)
  response = produce_response_from_reddit_data(llm, system_message, str(reddit_posts))

  return response