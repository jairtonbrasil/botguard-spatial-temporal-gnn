from enum import Enum
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class ActionType(str, Enum):
    POST = "POST"
    REPLY = "REPLY"
    RETWEET = "RETWEET"
    FOLLOW = "FOLLOW"

class UserAction(BaseModel):
    user_id: str
    target_id: Optional[str] = None
    action_type: ActionType
    content: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    true_label: int = Field(exclude=True)