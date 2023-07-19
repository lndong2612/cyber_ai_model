import jwt

from functools import wraps
from flask import request, jsonify
from config import settings
from typing import Dict


def decode_JWT(token:str) -> Dict:
    try:
        decode_token = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return decode_token 
    except:
        return None

def verify_jwt(jwtoken:str) -> bool:
    isTokenValid: bool = False
    try:
        payload = decode_JWT(jwtoken)
    except:
        payload = None
    if payload:
        isTokenValid = True
    return isTokenValid

def token_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get('X-Access-Token')
        if token is None or not verify_jwt(token):
            return jsonify(success=False, message="Token is missing or invalid"), 401
        return f(*args, **kwargs)
    return wrapper