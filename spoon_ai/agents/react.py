from spoon_ai.agents.base import BaseAgent
from abc import abstractmethod

class ReActAgent(BaseAgent):
    
    @abstractmethod
    async def think(self) -> bool:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    async def act(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    async def step(self) -> str:
        should_act = await self.think()
        if not should_act:
            return "Thinking completed. No action needed."
        
        return await self.act()