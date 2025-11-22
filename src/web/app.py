import uvicorn
from fastapi import FastAPI
from configuration.config import *
from web.schemas import Question, Answer
from web.service import ChatService

from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles





app = FastAPI()
app.mount("/templates", StaticFiles(directory=WEB_STATIC_DIR), name="templates")
service = ChatService()

@app.get("/")
def read_root():
    return RedirectResponse("/templates/index.html")

@app.post("/chat")
def read_item(question: Question) -> Answer:
    answer = service.chat(question)
    return Answer(message=answer)

if __name__ == '__main__':
    uvicorn.run('web.app:app', host='0.0.0.0', port=8000)
