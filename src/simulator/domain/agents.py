import random
import uuid
from typing import Optional
from shared.domain.models import UserAction, ActionType

class BaseAgent:
    """
    Base class representing a simulated social network agent.
    """

    def __init__(self, user_id: Optional[str] = None):
        """
        Initializes the agent with a unique user identifier.

        Args:
            user_id: Optional unique identifier for the user. If not provided,
                     a random UUID string is generated.
        """
        self.user_id = user_id or str(uuid.uuid4())

    def generate_action(self, content: str = "", target_id: Optional[str] = None) -> UserAction:
        """
        Generates a simulated social media action. This method must be implemented
        by subclasses to define specific agent behaviors.

        Args:
            content: Optional text content associated with the action.
            target_id: Optional target identifier for interactions like follow or reply.

        Returns:
            UserAction: The simulated user action model.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError


class NormalUserAgent(BaseAgent):
    """
    Simulates the behavior of a standard human social network user.
    """

    def generate_action(self, content: str = "", target_id: Optional[str] = None) -> UserAction:
        """
        Generates a simulated social media action typical of a human user.

        Human agents are configured with probability distributions that favor
        original posts over replies and following. The action is labeled
        with a true label of 0 (human).

        Args:
            content: Optional text content for the post or reply.
            target_id: Optional target identifier for follow or reply actions.

        Returns:
            UserAction: The generated human user action.
        """
        action_type = random.choices(
            [ActionType.POST, ActionType.REPLY, ActionType.FOLLOW],
            weights=[0.7, 0.2, 0.1]
        )[0]
        
        return UserAction(
            user_id=self.user_id,
            target_id=target_id if action_type != ActionType.POST else None,
            action_type=action_type,
            content=content if action_type in [ActionType.POST, ActionType.REPLY] else None,
            true_label=0
        )


class BotAgent(BaseAgent):
    """
    Simulates the behavior of an automated bot social network user.
    """

    def generate_action(self, content: str = "", target_id: Optional[str] = None) -> UserAction:
        """
        Generates a simulated social media action typical of an automated bot.

        Bot agents are configured with probability distributions that favor
        mass-following and replying over original posting. The action is
        labeled with a true label of 1 (bot).

        Args:
            content: Optional text content for the post or reply.
            target_id: Optional target identifier for follow or reply actions.

        Returns:
            UserAction: The generated bot user action.
        """
        action_type = random.choices(
            [ActionType.POST, ActionType.REPLY, ActionType.FOLLOW],
            weights=[0.1, 0.4, 0.5]
        )[0]
        
        return UserAction(
            user_id=self.user_id,
            target_id=target_id if action_type != ActionType.POST else None,
            action_type=action_type,
            content=content if action_type in [ActionType.POST, ActionType.REPLY] else None,
            true_label=1
        )