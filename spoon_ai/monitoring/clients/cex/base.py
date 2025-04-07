# spoon_ai/monitoring/clients/cex/base.py
import logging
from abc import abstractmethod
from typing import Dict, Any, List, Optional

from ..base import DataClient

logger = logging.getLogger(__name__)

class CEXClient(DataClient):
    """Centralized exchange client base class"""
    
    @abstractmethod
    def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """Get trading pair price"""
        pass
    
    @abstractmethod
    def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        """Get 24-hour statistics"""
        pass
    
    @abstractmethod
    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Any]:
        """Get K-line data"""
        pass
    
    @abstractmethod
    def get_server_time(self) -> int:
        """Get server time"""
        pass