import time
import random
import logging
from abc import ABC, abstractmethod
from typing import List, Generator, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MLPipeline_Phase2")

# Importing structural elements from Phase 1
@dataclass
class SensorReading:
    timestamp: float
    machine_id: str
    temperature: float
    vibration: float
    rotation_speed: float
    label: int = 0


class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, data: List[SensorReading]) -> List[SensorReading]:
        pass

class Standardizer(BaseTransformer):
    def transform(self, data: List[SensorReading]) -> List[SensorReading]:
        temps = [r.temperature for r in data]
        mean_temp = sum(temps) / len(temps) if temps else 90.0
        for reading in data:
            reading.temperature = reading.temperature - (mean_temp * 0.05)
        return data


# =====================================================================
# MILESTONE 3: Data Structures & Functional Abstraction (Weeks 5-7)
# =====================================================================

class FunctionalDataPipeline:
    """Manages stream processing using pipelines, functional principles, and generators."""
    def __init__(self, transformers: List[BaseTransformer]):
        self.transformers = transformers

    def stream_loader(self, dataset: List[SensorReading]) -> Generator[SensorReading, None, None]:
        """Generator performing memory-efficient streaming."""
        for reading in dataset:
            yield reading

    def filter_high_temperature(self, stream: Generator[SensorReading, None, None], limit: float) -> Generator[SensorReading, None, None]:
        """Functional filter abstraction implemented via Python generator expressions."""
        return (reading for reading in stream if reading.temperature > limit)

    def run_pipeline(self, raw_data: List[SensorReading]) -> List[SensorReading]:
        current_data = raw_data
        for transformer in self.transformers:
            current_data = transformer.transform(current_data)
        return current_data


# =====================================================================
# MILESTONE 4: Robustness, Exceptions & System Design Patterns (Weeks 8-10)
# =====================================================================

# --- Robustness & Fault Tolerance ---
class PIPELINE_EXCEPTION(Exception):
    """Base exception for all pipeline custom errors."""
    pass

class DataCorruptionError(PIPELINE_EXCEPTION):
    """Raised when data values breach physical safety rules."""
    pass

class ModelNotTrainedError(PIPELINE_EXCEPTION):
    """Raised if execution is triggered on an uninitialized model."""
    pass


# --- Design Pattern 1: Strategy Pattern ---
class PredictionStrategy(ABC):
    @abstractmethod
    def fit(self, features: List[List[float]], labels: List[int]) -> None:
        pass

    @abstractmethod
    def predict(self, feature_row: List[float]) -> int:
        pass


class ThresholdHeuristicStrategy(PredictionStrategy):
    """A deterministic machine learning heuristic model."""
    def __init__(self):
        self.is_trained = False
        self.temp_threshold = 0.0

    def fit(self, features: List[List[float]], labels: List[int]) -> None:
        failures = [features[i][0] for i in range(len(features)) if labels[i] == 1]
        self.temp_threshold = sum(failures) / len(failures) if failures else 90.0
        self.is_trained = True
        logger.info(f"Model strategy calibrated. Warning threshold: {self.temp_threshold:.2f}°C")

    def predict(self, feature_row: List[float]) -> int:
        if not self.is_trained:
            raise ModelNotTrainedError("Prediction failed: Model has not been trained yet!")
        return 1 if feature_row[0] > self.temp_threshold else 0


# --- Design Pattern 2: Observer Pattern ---
class PipelineObserver(ABC):
    @abstractmethod
    def update(self, event_type: str, data: Any) -> None:
        pass


class AlertSystem(PipelineObserver):
    """Triggers telemetry warnings in critical states."""
    def update(self, event_type: str, data: Any) -> None:
        if event_type == "FAILURE_DETECTED":
            logger.warning(f"⚠️  [ALERT SYSTEM]: High failure anomaly flag detected on entity: {data}")


# =====================================================================
# PHASE 2 EXECUTION HARNESS
# =====================================================================
if __name__ == "__main__":
    logger.info("=== STARTING PHASE 2 (MILESTONES 3 & 4) ===")
    
    # Milestone 3 Mock Dataset Load
    mock_data = [
        SensorReading(time.time(), "MACH_001", random.uniform(65, 110), random.uniform(2, 15), 1800, i % 5 == 0)
        for i in range(20)
    ]
    
    # Run the Pipeline
    data_pipeline = FunctionalDataPipeline([Standardizer()])
    processed_list = data_pipeline.run_pipeline(mock_data)
    
    # Milestone 4 Strategy and Observers Setup
    alert_notifier = AlertSystem()
    ml_strategy = ThresholdHeuristicStrategy()
    
    # Format mock features [temperature, vibration, rotation]
    features = [[r.temperature, r.vibration, r.rotation_speed] for r in processed_list]
    labels = [r.label for r in processed_list]
    
    # Train Heuristic Strategy
    ml_strategy.fit(features, labels)
    
    # Simulating standard inference + custom exception checks
    try:
        sample_feature = features[0]
        # Check for arbitrary data corruption bounds
        if sample_feature[0] < -50 or sample_feature[1] < 0:
            raise DataCorruptionError("Sensor hardware reported illegal physical values!")
            
        prediction = ml_strategy.predict(sample_feature)
        logger.info(f"Sample prediction result: {prediction}")
        if prediction == 1:
            alert_notifier.update("FAILURE_DETECTED", f"MACH_001 Temp: {sample_feature[0]:.2f}")
            
    except PIPELINE_EXCEPTION as e:
        logger.error(f"Gracefully caught Pipeline Exception: {e}")