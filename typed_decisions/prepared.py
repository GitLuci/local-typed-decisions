"""Explicit, process-local state handle. No global response cache or disk storage."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreparedState:
    # The owner validates the handle. Tensor storage is intentionally private;
    # this is not a security boundary against Python code inside the process.
    _owner: object = field(repr=False)
    _state_text: str = field(repr=False)
    _prefix_ids: tuple[int, ...] = field(repr=False)
    _layers: object = field(repr=False)
    _versions: tuple = field(repr=False)
    readout: str
    preparation_ms: float

    @property
    def prefix_tokens(self):
        return len(self._prefix_ids)

    @property
    def storage_bytes(self):
        return sum(t.numel()*t.element_size() for pair in self._layers.values() for t in pair)
