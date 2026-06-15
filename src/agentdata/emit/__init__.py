"""Training-ready output emitters + the factory that picks one by name."""

from __future__ import annotations

from .base import Emitter
from .chat import ChatEmitter, ShareGPTEmitter
from .devset import DevsetEmitter
from .dpo import DPOEmitter
from .easydataset import EasyDatasetEmitter
from .pretrain import PretrainEmitter
from .sft import SFTEmitter

_EMITTERS = {
    SFTEmitter.name: SFTEmitter,
    DPOEmitter.name: DPOEmitter,
    PretrainEmitter.name: PretrainEmitter,
    ChatEmitter.name: ChatEmitter,
    ShareGPTEmitter.name: ShareGPTEmitter,
    EasyDatasetEmitter.name: EasyDatasetEmitter,
    DevsetEmitter.name: DevsetEmitter,
}


def build_emitter(name: str) -> Emitter:
    """Select an emitter by Recipe.emit / --emit key."""
    cls = _EMITTERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown emit format {name!r}. Expected one of: {', '.join(sorted(_EMITTERS))}."
        )
    return cls()


__all__ = ["Emitter", "build_emitter"]
