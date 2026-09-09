from .schemas import ChangeEvidence


class DeltaVLMAdapter:
    """
    Application-level adapter for DeltaVLM.

    The SATQUERY application communicates with DeltaVLM through
    this interface instead of depending directly on DeltaVLM
    implementation details.
    """

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """Load the DeltaVLM model and checkpoint."""
        raise NotImplementedError(
            "DeltaVLM model loading will be implemented "
            "when the GPU inference environment is connected."
        )

    def analyze(self, image_t1, image_t2, query) -> ChangeEvidence:
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
        """
        raise NotImplementedError(
            "DeltaVLM inference will be implemented next."
        )
