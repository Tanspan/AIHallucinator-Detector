"""core/base_agent.py"""
import logging
from abc import ABC

class BaseAgent(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def log(self, msg: str, level: str = "info") -> None:
        getattr(self.logger, level)(f"[{self.name}] {msg}")
