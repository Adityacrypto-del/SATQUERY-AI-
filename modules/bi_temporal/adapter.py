import logging
from typing import Optional

from .schemas import ChangeEvidence

logger = logging.getLogger(__name__)


class DeltaVLMAdapter:
    """
    Application-level adapter for DeltaVLM.

    The SATQUERY application communicates with DeltaVLM through
    this interface instead of depending directly on DeltaVLM
    implementation details.

    On environments without NVIDIA CUDA (e.g. Apple Silicon),
    the adapter reports itself as unavailable and fails
    gracefully with a clear message.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self._available: Optional[bool] = None
        self._unavailable_reason: str = ""

    @property
    def is_available(self) -> bool:
        """Whether DeltaVLM can run on this environment.

        Checks for CUDA availability on first access and
        caches the result.
        """
        if self._available is not None:
            return self._available

        try:
            import torch

            if not torch.cuda.is_available():
                self._available = False
                self._unavailable_reason = (
                    "DeltaVLM requires an NVIDIA GPU with CUDA. "
                    "Current device does not have CUDA available."
                )
                return False

        except ImportError:
            self._available = False
            self._unavailable_reason = (
                "PyTorch is not installed in this environment."
            )
            return False

        self._available = True
        return True

    def load_model(self) -> None:
        """Load the DeltaVLM model and checkpoint.

        Raises
        ------
        RuntimeError
            If the environment does not support DeltaVLM
            (no CUDA, missing dependencies, etc.).
        """
        if not self.is_available:
            logger.warning(
                "DeltaVLM unavailable: %s",
                self._unavailable_reason,
            )
            raise RuntimeError(self._unavailable_reason)

        # Actual model loading will be implemented when the
        # GPU inference environment is connected.
        raise RuntimeError(
            "DeltaVLM model loading is not yet implemented. "
            "The interface is ready — connect a CUDA environment "
            "and provide the checkpoint path via model_path."
        )

    def analyze(
        self,
        image_t1,
        image_t2,
        query: str,
    ) -> ChangeEvidence:
        """
        Analyze two spatially corresponding images.

        Parameters
        ----------
        image_t1:
            Earlier observation.

        image_t2:
            Later observation.

        query:
            Natural-language question about temporal change.

        Raises
        ------
        RuntimeError
            If DeltaVLM is not available or not loaded.
        """
        if not self.is_available:
            raise RuntimeError(
                f"DeltaVLM is not available: {self._unavailable_reason}"
            )

        if self.model is None:
            raise RuntimeError(
                "DeltaVLM model not loaded. Call load_model() first."
            )

        # Actual inference will be implemented when the GPU
        # environment is connected.
        raise RuntimeError(
            "DeltaVLM inference is not yet implemented."
        )

