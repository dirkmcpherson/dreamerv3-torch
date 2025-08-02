# import dataclass
from dataclasses import dataclass

@dataclass(frozen=True)
class Constants:
    COMPILE: bool = True

STATIC_CONSTANTS = Constants()