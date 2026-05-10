import time
import random
import logging
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MLPipeline_Phase1")

# =====================================================================
# MILESTONE 1: Object Foundations & System Modeling (Weeks 1-2)
# =====================================================================

@dataclass
class SensorReading:
    """Represents a single multidimensional data observation from a machine."""
    timestamp: float
    machine_id: str
    temperature: float    # in °C
    vibration: float      # in mm/s
    rotation_speed: float  # in RPM
    label: int = 0        # 0 = Normal, 1 = Failure


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
        # Generating synthetic but highly realistic physical telemetry measurements
        self._data = [
            SensorReading(time.time() - (i * 10), "MACH_001", 
                          random.uniform(65.0, 110.0), 
                          random.uniform(2.0, 15.0), 
                          random.uniform(1200, 3200),
                          label=1 if random.random() > 0.85 else 0)
            for i in range(50)
        ]
        logger.info(f"Loaded {len(self._data)} sensor records into the dataset.")
        return self._data


# =====================================================================
# MILESTONE 2: Object Interaction & Control Logic (Weeks 3-4)
# =====================================================================

class BaseTransformer(ABC):
    """Polymorphic Base class for all pipeline transformation steps."""
    @abstractmethod
    def transform(self, data: List[SensorReading]) -> List[SensorReading]:
        pass


class Standardizer(BaseTransformer):
    """Polymorphic transformer that scales sensor readings dynamically."""
    def transform(self, data: List[SensorReading]) -> List[SensorReading]:
        if not data:
            return data
        
        temps = [r.temperature for r in data]
        mean_temp = sum(temps) / len(temps)
        
        logger.info(f"Standardizing dataset against Mean Temperature: {mean_temp:.2f}°C")
        for reading in data:
            reading.temperature = reading.temperature - (mean_temp * 0.05)
        return data


class OutlierFilter(BaseTransformer):
    """Filters anomalous extreme system noise before modeling."""
    def transform(self, data: List[SensorReading]) -> List[SensorReading]:
        cleaned = [r for r in data if r.vibration < 14.5]
        logger.info(f"Outlier removal completed: {len(data) - len(cleaned)} noise entries purged.")
        return cleaned


# =====================================================================
# PHASE 1 EXECUTION HARNESS
# =====================================================================
if __name__ == "__main__":
    logger.info("=== STARTING PHASE 1 (MILESTONES 1 & 2) ===")
    
    # Milestone 1: Load Initial Data
    dataset_loader = RealWorldSensorDataset()
    raw_sensor_data = dataset_loader.load_data()
    
    # Milestone 2: Execute PolyMorphic Interacting Transforms
    transformers = [Standardizer(), OutlierFilter()]
    
    processed_data = raw_sensor_data
    for transformer in transformers:
        processed_data = transformer.transform(processed_data)
        
    logger.info(f"Phase 1 Run Complete. Processed record count: {len(processed_data)}")