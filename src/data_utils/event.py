import typing as tp
from dataclasses import dataclass, fields

@dataclass
class Event:
    """
    Base class for all event kinds with the bare minimum common fields.

    If the event is instantiated with `from_dict()`, additional non-required fields that are
    provided will be ignored instead of causing an error.
    """
    start: float
    duration: float
    modality: tp.Optional[str]  # Move to keyword arguments when updating to 3.10
    language: tp.Optional[str]  # Move to keyword arguments when updating to 3.10
    # See https://www.trueblade.com/blogs/news/python-3-10-new-dataclass-features

    def __post_init__(self):
        if self.duration < 0:
            raise ValueError("Negative durations are not allowed for events.")
        # self._sample_rate: tp.Optional[utils.Frequency] = None  # XXX Necessary?

    @classmethod
    def from_dict(cls, row: dict) -> "Event":
        """Create event from dictionary while ignoring extra parameters.
        """
        return cls(**{k: v for k, v in row.items() if k in [f.name for f in fields(cls)]})

    @classmethod
    def _kind(cls) -> str:
        """Convenience method to get the name from the class."""
        return cls.__name__.lower()

    @property
    def kind(self) -> str:
        return self.__class__._kind()

    @property
    def stop(self) -> float:
        return self.start + self.duration
    

@dataclass
class Block(Event):
    uid: str

    def __post_init__(self):
        super().__post_init__()
        self.uid = str(self.uid)