import yaml
import os

from src.review_analysis import fetch_data, load_sentiment_model, load_anthropic_model, augment_prompt, get_auth_token, get_reddit_posts_and_comments, generate_review

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

@app.post("/get_output/")
async def get_output(file: UploadFile = File(...)):
    try:
        with open('vars.yaml', 'r') as file:
            config = yaml.safe_load(file)
            print(config)

                # Use the parsed parameters and get results
            
            params = fetch_data(config)
            print(params['anthropic_api_key'])
            results = generate_review(
                params['sentiment_model'],
                params['llm_model'],
                params['llm_temperature'],
                params['anthropic_api_key'],
                params['reddit_search_query'],
                params['reddit_credentials'],
                params['search_query_parameters'],
                params['comment_parameters'],
                params['system_message'],
                params['subreddit'],
            )

            print("Results complied")
            return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

