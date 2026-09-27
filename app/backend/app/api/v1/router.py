from fastapi import APIRouter

from app.features.audit.router import router as audit_router
from app.features.auth.router import router as auth_router
from app.features.cameras.router import router as cameras_router
from app.features.candidates.router import router as candidates_router
from app.features.events.router import router as events_router
from app.features.health.router import router as health_router
from app.features.media.router import router as media_router
from app.features.monitoring.router import router as monitoring_router
from app.features.monitoring.rtsp_router import router as rtsp_router
from app.features.rooms.router import router as rooms_router
from app.features.sessions.router import router as sessions_router
from app.features.users.router import router as users_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health_router)
api_router.include_router(auth_router)
api_router.include_router(cameras_router)
api_router.include_router(users_router)
api_router.include_router(audit_router)
api_router.include_router(rooms_router)
api_router.include_router(candidates_router)
api_router.include_router(media_router)
api_router.include_router(sessions_router)
api_router.include_router(monitoring_router)
api_router.include_router(events_router)
api_router.include_router(rtsp_router)
