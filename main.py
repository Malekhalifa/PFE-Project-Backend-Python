from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db import connect, disconnect
from routes import router


app = FastAPI()


@app.on_event("startup")
async def startup() -> None:
    await connect()


@app.on_event("shutdown")
async def shutdown() -> None:
    await disconnect()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)
