from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional
import joblib
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from config import MONGODB_URI, DATABASE_NAME, PREDICTION_COLLECTION, USER_COLLECTION
from model import WDBCFeatures, User
import logging
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Initialize FastAPI app
app = FastAPI(title="Breast Cancer Prediction API",)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Load the pre-trained model
try:
    model = joblib.load("model_training/rf_best_model.joblib")
    scaler = joblib.load("model_training/scaler.joblib")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to load model") from e

# Connect to MongoDB
try:
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    predictions_collection = db[PREDICTION_COLLECTION]
    users_collection = db[USER_COLLECTION]
    logger.info("Connected to MongoDB successfully")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {str(e)}")
    raise RuntimeError("Failed to connect to MongoDB") from e

async def get_current_user(request: Request):
    credentials: HTTPAuthorizationCredentials = await security(request)
    # In a real application, you would validate the token here
    # For simplicity, we'll just use the token as the username
    return {"username": credentials.credentials}

@app.post("/register/")
async def register_user(user: User):
    try:
        # Check if user already exists
        if users_collection.find_one({"username": user.username}):
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Insert new user
        user_dict = user.dict()
        user_dict["created_at"] = datetime.utcnow()
        users_collection.insert_one(user_dict)
        
        return {"message": "User registered successfully"}
    except Exception as e:
        logger.error(f"User registration error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-wdbc", response_model=Dict[str, Any])
async def predict_wdbc(features: WDBCFeatures, request: Request):
    """
    Predict whether a tumor is malignant or benign based on WDBC features.
    
    Returns:
        dict: {
            "prediction": "malignant" or "benign",
            "confidence": probability value (0.0-1.0)
        }
    """
    try:
        # Get current user (simplified authentication)
        current_user = await get_current_user(request)
        username = current_user["username"]
        
        # Check if user exists
        user = users_collection.find_one({"username": username})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Convert input features to numpy array for prediction
        input_data = np.array([
            features.radius_mean,
            features.texture_mean,
            features.perimeter_mean,
            features.area_mean,
            features.smoothness_mean,
            features.compactness_mean,
            features.concavity_mean,
            features.concave_points_mean,
            features.symmetry_mean,
            features.fractal_dimension_mean,
            features.radius_se,
            features.texture_se,
            features.perimeter_se,
            features.area_se,
            features.smoothness_se,
            features.compactness_se,
            features.concavity_se,
            features.concave_points_se,
            features.symmetry_se,
            features.fractal_dimension_se,
            features.radius_worst,
            features.texture_worst,
            features.perimeter_worst,
            features.area_worst,
            features.smoothness_worst,
            features.compactness_worst,
            features.concavity_worst,
            features.concave_points_worst,
            features.symmetry_worst,
            features.fractal_dimension_worst
        ]).reshape(1, -1)
        
        # Make prediction
        prediction_proba = model.predict_proba(input_data)
        confidence = float(np.max(prediction_proba))
        prediction = "malignant" if model.predict(input_data)[0] == 1 else "benign"
        
        # Prepare record for MongoDB
        record = {
            "username": username,
            "input_features": features.dict(),
            "prediction": prediction,
            "confidence": confidence,
            "timestamp": datetime.utcnow()
        }
        
        # Insert record into MongoDB
        try:
            predictions_collection.insert_one(record)
            logger.info(f"Prediction saved to MongoDB for user {username}")
            
            # Update user's history
            users_collection.update_one(
                {"username": username},
                {"$push": {"prediction_history": record["_id"]}},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {str(e)}")
            # Continue even if MongoDB fails
        
        return {
            "prediction": prediction,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/user/history/")
async def get_user_history(request: Request):
    """
    Get prediction history for the current user
    """
    try:
        # Get current user
        current_user = await get_current_user(request)
        username = current_user["username"]
        
        # Get user's prediction history
        user = users_collection.find_one(
            {"username": username},
            {"prediction_history": 1}
        )
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get all prediction records
        history_ids = user.get("prediction_history", [])
        predictions = list(predictions_collection.find(
            {"_id": {"$in": history_ids}},
            {"_id": 0, "input_features": 1, "prediction": 1, "confidence": 1, "timestamp": 1}
        ).sort("timestamp", -1))
        
        return {"history": predictions}
    except Exception as e:
        logger.error(f"Error getting user history: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))