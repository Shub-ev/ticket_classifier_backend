from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from endpoint.triage import triage_router


# Development frontend origin; adjust deliberately when deploying the API.
origins = [
    "http://localhost:5173",
]

def init_server():
    """Build the FastAPI application and register its router and CORS policy."""
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
    """Return a lightweight service health response."""
    return {"message": "Hello World!"}