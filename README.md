# Reddit API Integration with LLM Recommendations

## Overview
This API utilizes the Reddit API to retrieve posts and comments relevant to a given user query. It then uses Anthropic's  large language model (LLM) to produce recommendations with citations. The API is developed using FastAPI and is launched using Uvicorn.

## Requirements
- Python 3.8+
- `langchain==0.2.5`
- `langchain-community==0.2.5`
- `langchain-anthropic==0.1.15`
- `fastapi==0.111.0`
- `uvicorn==0.30.1`
- `pyyaml==6.0.1`
- `torch==2.2.0`
- `transformers==4.41.2`

## Setup

### .env File
Create a `.env` file in the root directory of your project with the following headings and provide the necessary credentials:
```
ANTHROPIC_API_KEY=KEY
REDDIT_CLIENT_ID=ID
REDDIT_CLIENT_SECRET=SECRET
REDDIT_USERNAME=USERNAME
REDDIT_PASSWORD=PASSWORD
REDDIT_USER_AGENT=AGENT
```

### Dependencies
Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Configuration
Modify these fields to control the behavior of the API. Here are the definitions of each field:

- **anthropic_api_key**: (read from env file) API key for accessing the Anthropic service.
- **llm_model**: The LLM model to be used, specified by name (e.g., "claude-3-opus-20240229").
- **llm_temperature**: The temperature setting for the LLM, controlling the randomness of outputs.
- **sentiment_model**: The sentiment analysis model to be used, specified by name (e.g., "distilbert-base-uncased-finetuned-sst-2-english").
- **reddit_credentials**: (read from env file) A dictionary containing the Reddit API credentials:
  - **client_id**: The client ID for Reddit API.
  - **client_secret**: The client secret for Reddit API.
  - **username**: The Reddit username.
  - **password**: The Reddit password.
  - **user_agent**: The user agent string for the Reddit API.
- **reddit_search_query**: The query string to search for relevant Reddit posts (e.g., "what is the most reliable compact, non-luxury car?").
- **search_query_parameters**: A dictionary containing parameters for the Reddit search query:
  - **sort**: The sorting method for search results (e.g., 'relevance').
  - **limit**: The maximum number of search results to retrieve.
- **subreddit**: The specific subreddit to search within (e.g., 'whatcarshouldIbuy').
- **comment_parameters**: A dictionary containing parameters for retrieving comments:
  - **depth**: The depth of comment threads to retrieve.
  - **limit**: The maximum number of comments per post.
  - **sort**: The sorting method for comments (e.g., 'top').
- **system_message**: A placeholder for any system message for the LLM to create the final summary based on the Reddit posts and comments. Modify this to control the final output.

### Launching API
Run this in the command line to launch the API. 
```bash
uvicorn main:app
```

### Output URL
The input YAML file takes in the config file and produces the summary in the form of a string output. The post request can be accessed at:
``` 
http://localhost:8000/get_output/
```
