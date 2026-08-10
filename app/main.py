from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="AI microservice for SmartOrg — handles document processing and RAG-based chat",
)

cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")] if settings.CORS_ORIGINS != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/ai")

@app.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "status":  "running",
        "port":    settings.PORT,
        "docs":    "/docs",
    }