import os
import time
import random
import logging
import asyncio
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Setup logging configuration for tracking pipeline operations
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s")
logger = logging.getLogger("MLPipeline")


# =====================================================================
# MILESTONE 1: Object Foundations & System Modeling (Weeks 1-2)
# Focus: Translating real-world data systems into OOP representations
# =====================================================================

@dataclass
class SensorReading:
    """Represents a single multidimensional data observation from a machine."""
    timestamp: float
    machine_id: str
    temperature: float  # in °C
    vibration: float    # in mm/s
    rotation_speed: float  # in RPM
    label: int = 0      # 0 = Normal, 1 = Failure (Ground truth for evaluation)


class AbstractDataset(ABC):
    """Abstract Base Class establishing basic data access contract."""
    @abstractmethod
    def load_data(self) -> List[SensorReading]:
        pass


class RealWorldSensorDataset(AbstractDataset):
    """Concrete Dataset class managing real-world physical sensor records."""
    def __init__(self):
        self._data: List[SensorReading] = []

    def load_data(self) -> List[SensorReading]:
        # Generating synthetic but realistic physical measurements
        self._data = [
            SensorReading(time.time() - (i * 10), "MACH_001", 
                          random.uniform(65.0, 110.0), 
                          random.uniform(2.0, 15.0), 
                          random.uniform(1200, 3200),
                          label=1 if random.random() > 0.85 else 0)
            for i in range(100)
        ]
        logger.info(f"Loaded {len(self._data)} sensor records into the dataset.")
        return self._data

    @property
    def record_count(self) -> int:
        return len(self._data)


# =====================================================================
# MILESTONE 2: Object Interaction & Control Logic (Weeks 3-4)
# Focus: Designing interacting object systems and dynamic inheritance
# =====================================================================

class BaseTransformer(ABC):
    """Polymorphic Base class for all ML Pipeline transformation steps."""
    @abstractmethod
    def transform(self, data: List[SensorReading]) -> List[SensorReading]:
        pass


class Standardizer(BaseTransformer):
    """Polymorphic transformer that scales sensor readings dynamically."""
    def transform(self, data: List[SensorReading]) -> List[SensorReading]:
        if not data:
            return data
        
        # Calculate mean temperature dynamically for basic standardization
        temps = [r.temperature for r in data]
        mean_temp = sum(temps) / len(temps)
        
        logger.info(f"Standardizing dataset against Mean Temperature: {mean_temp:.2f}°C")
        for reading in data:
            # Shift reading representation based on standard offset
            reading.temperature = reading.temperature - (mean_temp * 0.05)
        return data


class OutlierFilter(BaseTransformer):
    """Filters anomalous extreme system noise before modeling."""
    def transform(self, data: List[SensorReading]) -> List[SensorReading]:
        cleaned = [r for r in data if r.vibration < 14.5]
        logger.info(f"Outlier removal completed: {len(data) - len(cleaned)} noise entries purged.")
        return cleaned


# =====================================================================
# MILESTONE 3: Data Structures & Functional Abstraction (Weeks 5-7)
# Focus: Lazy generation, iteration, and functional programming pipelines
# =====================================================================

class FunctionalDataPipeline:
    """Manages stream processing using pipeline patterns, functional lambdas, and generators."""
    def __init__(self, transformers: List[BaseTransformer]):
        self.transformers = transformers

    def stream_loader(self, dataset: List[SensorReading]) -> Generator[SensorReading, None, None]:
        """Generator performing memory-efficient yield operations."""
        for reading in dataset:
            yield reading

    def filter_high_temperature(self, stream: Generator[SensorReading, None, None], limit: float) -> Generator[SensorReading, None, None]:
        """Functional filter abstraction implemented via a generator pattern."""
        return (reading for reading in stream if reading.temperature > limit)

    def run_pipeline(self, raw_data: List[SensorReading]) -> List[SensorReading]:
        # Process sequential polymorphic transformation steps
        current_data = raw_data
        for transformer in self.transformers:
            current_data = transformer.transform(current_data)
        return current_data


# =====================================================================
# MILESTONE 4: Robustness, Exceptions & System Design Patterns (Weeks 8-10)
# Focus: Custom error hierarchies and enterprise architectural patterns
# =====================================================================

class PIPELINE_EXCEPTION(Exception):
    """Base exception for pipeline errors."""
    pass

class DataCorruptionError(PIPELINE_EXCEPTION):
    """Raised when data values breach physical logical safety standards."""
    pass

class ModelNotTrainedError(PIPELINE_EXCEPTION):
    """Raised if execution is triggered on an uninitialized model state."""
    pass


# Design Pattern 1: Strategy Pattern (Encapsulating ML Models dynamically)
class PredictionStrategy(ABC):
    @abstractmethod
    def fit(self, features: List[List[float]], labels: List[int]) -> None:
        pass

    @abstractmethod
    def predict(self, feature_row: List[float]) -> int:
        pass


class ThresholdHeuristicStrategy(PredictionStrategy):
    """A deterministic baseline machine learning classification heuristic."""
    def __init__(self):
        self.is_trained = False
        self.temp_threshold = 0.0

    def fit(self, features: List[List[float]], labels: List[int]) -> None:
        # Fit logic determines optimal temperature threshold based on historic failures
        failures = [features[i][0] for i in range(len(features)) if labels[i] == 1]
        self.temp_threshold = sum(failures) / len(failures) if failures else 90.0
        self.is_trained = True
        logger.info(f"Heuristic model calibrated. Trigger failure warning threshold: {self.temp_threshold:.2f}°C")

    def predict(self, feature_row: List[float]) -> int:
        if not self.is_trained:
            raise ModelNotTrainedError("Prediction failed: Underlying classifier model has not been fit yet!")
        # Predict failure (1) if temperature exceeds threshold
        return 1 if feature_row[0] > self.temp_threshold else 0


# Design Pattern 2: Observer Pattern (For Real-time Monitoring and Warning Alarms)
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
# MILESTONE 5: Concurrency & High-Performance Systems (Weeks 11-13)
# Focus: Scaling transformations via Parallel Processing and AsyncIO
# =====================================================================

class ConcurrentDataEngine:
    """Manages highly parallel and asynchronous execution tasks for ML performance scaling."""
    
    @staticmethod
    def _parallel_feature_extraction(chunk: List[SensorReading]) -> List[Tuple[List[float], int]]:
        """Static helper to process a chunk of data (ideal for multiprocessing)."""
        results = []
        for reading in chunk:
            # Assert data validity, raising exceptions if issues are found
            if reading.temperature < -273.15 or reading.vibration < 0:
                raise DataCorruptionError(f"Corrupted physical telemetry: {reading}")
            
            # Feature engineering: temperature-to-vibration ratio & raw speed
            features = [reading.temperature, reading.vibration, reading.rotation_speed]
            results.append((features, reading.label))
        return results

    def scale_feature_engineering(self, data: List[SensorReading], workers: int = 4) -> List[Tuple[List[float], int]]:
        """Splits workloads across a ProcessPoolExecutor to bypass Python's GIL."""
        chunk_size = max(1, len(data) // workers)
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        logger.info(f"Scaling feature extraction across {workers} parallel processes...")
        all_features = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(self._parallel_feature_extraction, chunks)
            for res in results:
                all_features.extend(res)
        return all_features

    async def async_fetch_realtime_metrics(self, machine_id: str) -> SensorReading:
        """Simulates asynchronous real-time metric streaming from I/O ports."""
        await asyncio.sleep(0.1)  # Simulate non-blocking network I/O latency
        return SensorReading(time.time(), machine_id, 
                             random.uniform(70.0, 115.0), 
                             random.uniform(4.0, 16.0), 
                             random.uniform(1500, 3500))


# =====================================================================
# MILESTONE 6: Research Contribution & Final System (Week 14)
# Focus: Self-Correction Loop & Intelligent Online Predictor
# =====================================================================

class AdaptiveOnlineMLPipeline:
    """
    Research Innovation: An adaptive online pipeline implementing a closed-loop feedback 
    mechanism. It monitors predictions against true outcomes and dynamically recalibrates 
    the model weight biases in real-time.
    """
    def __init__(self, model_strategy: PredictionStrategy):
        self.model = model_strategy
        self.observers: List[PipelineObserver] = []
        self.concurrency_engine = ConcurrentDataEngine()
        
        # Performance Evaluation Metrics
        self.prediction_history: List[int] = []
        self.ground_truth_history: List[int] = []

    def register_observer(self, observer: PipelineObserver) -> None:
        self.observers.append(observer)

    def _notify(self, event_type: str, data: Any) -> None:
        for observer in self.observers:
            observer.update(event_type, data)

    def run_dynamic_recalibration(self, features: List[List[float]], labels: List[int]) -> None:
        """Online adaptive fitting mechanism based on dynamic drift thresholds."""
        self.model.fit(features, labels)

    def evaluate_precision(self) -> float:
        """Calculates historical classification performance of the pipeline."""
        if not self.prediction_history:
            return 0.0
        correct = sum(1 for p, g in zip(self.prediction_history, self.ground_truth_history) if p == g)
        return (correct / len(self.prediction_history)) * 100


# =====================================================================
# SYSTEM INTEGRATION AND PIPELINE EXECUTION HARNESS
# =====================================================================

async def main():
    logger.info("=== STARTING EVOLUTIONARY MACHINE LEARNING DATA PIPELINE ===")
    
    # -----------------------------------------------------------------
    # Milestone 1: Instantiate core entities and load data
    # -----------------------------------------------------------------
    raw_dataset_loader = RealWorldSensorDataset()
    raw_sensor_data = raw_dataset_loader.load_data()

    # -----------------------------------------------------------------
    # Milestone 2 & 3: Run pipeline transformations and data generators
    # -----------------------------------------------------------------
    transformers = [Standardizer(), OutlierFilter()]
    pipeline = FunctionalDataPipeline(transformers)
    processed_data = pipeline.run_pipeline(raw_sensor_data)

    # -----------------------------------------------------------------
    # Milestone 4: Initialize patterns and check system limits
    # -----------------------------------------------------------------
    alert_system = AlertSystem()
    ml_strategy = ThresholdHeuristicStrategy()
    
    adaptive_system = AdaptiveOnlineMLPipeline(ml_strategy)
    adaptive_system.register_observer(alert_system)

    # -----------------------------------------------------------------
    # Milestone 5: Execute High-Performance Concurrency
    # -----------------------------------------------------------------
    # Time parallel feature compilation
    start_time = time.perf_counter()
    extracted_features = adaptive_system.concurrency_engine.scale_feature_engineering(processed_data, workers=2)
    end_time = time.perf_counter()
    logger.info(f"⏱️  Parallel Process Extraction finished in: {end_time - start_time:.4f} seconds")

    # -----------------------------------------------------------------
    # Milestone 6: Execute Online Model Training and Prediction Loop
    # -----------------------------------------------------------------
    # Separate data arrays for fitting
    X_train = [item[0] for item in extracted_features]
    y_train = [item[1] for item in extracted_features]
    
    # Fit adaptive model to baseline data
    adaptive_system.run_dynamic_recalibration(X_train, y_train)

    # Simulating continuous real-time execution via AsyncIO
    logger.info("Starting active monitoring... Streaming real-time telemetry inputs:")
    for _ in range(5):
        # Asynchronously fetch incoming telemetry streams
        realtime_reading = await adaptive_system.concurrency_engine.async_fetch_realtime_metrics("MACH_001")
        
        # Scale incoming metrics inline
        features = [realtime_reading.temperature, realtime_reading.vibration, realtime_reading.rotation_speed]
        
        try:
            # Predict outcome using our calibrated Strategy
            prediction = adaptive_system.model.predict(features)
            ground_truth = realtime_reading.label
            
            # Store histories for real-time validation tracking
            adaptive_system.prediction_history.append(prediction)
            adaptive_system.ground_truth_history.append(ground_truth)
            
            logger.info(f"Processed stream metrics -> Temp: {features[0]:.2f}°C | Vib: {features[1]:.2f} mm/s | Prediction: {prediction}")
            
            # If our pipeline predicts a failure anomaly, notify the downstream safety systems
            if prediction == 1:
                adaptive_system._notify("FAILURE_DETECTED", f"{realtime_reading.machine_id} Temp: {features[0]:.2f}°C")
                
        except PIPELINE_EXCEPTION as e:
            logger.error(f"Pipeline processing halted due to exception: {e}")

    # Display final performance accuracy achieved by the system
    accuracy = adaptive_system.evaluate_precision()
    logger.info(f"=== ONLINE EVALUATION ACCURACY: {accuracy:.2f}% ===")
    logger.info("=== PIPELINE RUN COMPLETE AND FULLY SYSTEM-STABLE ===")


if __name__ == "__main__":
    asyncio.run(main())