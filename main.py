from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from endpoint.triage import triage_router

origins = [
    "http://localhost:5173",
]

def init_server():
    app = FastAPI(title="Ticket Classification API")
    app.include_router(triage_router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app

app = init_server()

@app.get("/")
def main():
    return {"message": "Hello World!"}