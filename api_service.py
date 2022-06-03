from fastapi import FastAPI, Request
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi.exceptions import HTTPException
from asyncio import Lock, sleep
from starlette.status import HTTP_503_SERVICE_UNAVAILABLE
import subprocess
import json
import requests
from inference import do_everything, generate_ner_query_output

lock = Lock()
app = FastAPI()

@app.post("/ner_point")
async def ner(request: Request):
    context = await request.json()

    # TODO
    # Add pydantic check here

    linked_entities, relations = do_everything(context)
    json_response = generate_ner_query_output(linked_entities)

    # json_response = do_everything(context)
    
    return json_response
