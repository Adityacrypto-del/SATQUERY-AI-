from .evaluate_rsvqa import evaluate as evaluate_vqa
from .evaluate_vrsbench import evaluate as evaluate_captioning

__all__ = ["evaluate_vqa", "evaluate_captioning"]
