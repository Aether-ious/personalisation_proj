from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import RecSysService
from contextlib import asynccontextmanager

# Global Service Variable
rec_service = None

# Lifespan event to load model ONCE at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rec_service
    print("🤖 Initializing Recommendation Engine...")
    try:
        rec_service = RecSysService(config_path="config.yaml")
        print("✅ Service Ready!")
    except Exception as e:
        print(f"❌ Failed to load service: {e}")
    yield
    print("🛑 Shutting down...")

app = FastAPI(lifespan=lifespan, title="ShopSmart API")

class UserRequest(BaseModel):
    user_id: int
    k: int = 5

@app.get("/")
def home():
    return {"message": "ShopSmart Recommendation API is Running"}

@app.post("/recommend")
async def recommend(payload: UserRequest):
    if rec_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    try:
        recs = rec_service.get_recommendations(user_id=payload.user_id, k=payload.k)
        
        if not recs:
            return {"message": "User not found or Cold Start", "default_items": [1, 2, 3]}

        return {
            "user_id": payload.user_id,
            "recommendations": recs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))