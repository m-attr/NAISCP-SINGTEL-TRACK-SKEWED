from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    id_column: str = "CustomerID"
    time_column: str = "Month"
    target_column: str = "ChurnStatus"


PIPELINE_CONFIG = PipelineConfig()
