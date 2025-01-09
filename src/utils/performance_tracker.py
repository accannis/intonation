"""
Utility for tracking performance metrics
"""

import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class StageMetrics:
    """Metrics for a single stage"""
    start_time: float
    end_time: float = 0
    duration: float = 0

class PerformanceTracker:
    """Tracks performance metrics for different stages"""
    
    def __init__(self, name: str = ""):
        """Initialize tracker
        
        Args:
            name: Optional name for this tracker, used in logging
        """
        self.name = name
        self.stages: Dict[str, StageMetrics] = {}
        self.current_stage: Optional[str] = None
        self.logger = logging.getLogger(f"performance_tracker.{name}" if name else "performance_tracker")
        
    def _format_duration(self, duration: float) -> str:
        """Format duration for logging
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Formatted string with appropriate units
        """
        if duration < 0.001:  # Less than 1ms
            return f"{duration*1_000_000:.2f}μs"
        elif duration < 1.0:  # Less than 1s
            return f"{duration*1_000:.2f}ms"
        else:
            return f"{duration:.3f}s"
        
    @contextmanager
    def track_stage(self, stage_name: str):
        """Context manager for tracking a stage
        
        Args:
            stage_name: Name of the stage to track
        """
        try:
            self.start_stage(stage_name)
            yield
        finally:
            self.end_stage(stage_name)
            
    def start_stage(self, stage_name: str):
        """Start tracking a stage
        
        Args:
            stage_name: Name of the stage to track
        """
        self.current_stage = stage_name
        self.stages[stage_name] = StageMetrics(start_time=time.time())
        self.logger.debug(f"Starting stage: {stage_name}")
        
    def end_stage(self, stage_name: str):
        """End tracking a stage
        
        Args:
            stage_name: Name of the stage to end
        """
        if stage_name in self.stages:
            metrics = self.stages[stage_name]
            metrics.end_time = time.time()
            metrics.duration = metrics.end_time - metrics.start_time
            
            # Log completion with duration
            duration_str = self._format_duration(metrics.duration)
            self.logger.info(f"Completed stage: {stage_name} - Duration: {duration_str}")
            
        if self.current_stage == stage_name:
            self.current_stage = None
            
    def get_stage_duration(self, stage_name: str) -> float:
        """Get duration of a stage in seconds
        
        Args:
            stage_name: Name of stage to get duration for
            
        Returns:
            Duration in seconds, or 0 if stage not found
        """
        if stage_name in self.stages:
            return self.stages[stage_name].duration
        return 0
        
    def get_all_metrics(self) -> Dict[str, float]:
        """Get all stage durations
        
        Returns:
            Dict mapping stage names to durations in seconds
        """
        return {name: metrics.duration for name, metrics in self.stages.items()}
        
    def log_summary(self):
        """Log a summary of all stage durations"""
        if not self.stages:
            self.logger.info("No performance metrics recorded")
            return
            
        # Calculate total duration
        total_duration = sum(metrics.duration for metrics in self.stages.values())
        
        # Log summary header
        summary_header = f"Performance Summary"
        if self.name:
            summary_header += f" for {self.name}"
        self.logger.info("-" * 50)
        self.logger.info(summary_header)
        self.logger.info("-" * 50)
        
        # Log each stage
        for stage_name, metrics in sorted(self.stages.items(), key=lambda x: x[1].duration, reverse=True):
            duration_str = self._format_duration(metrics.duration)
            percentage = (metrics.duration / total_duration) * 100 if total_duration > 0 else 0
            self.logger.info(f"{stage_name:.<40} {duration_str:>10} ({percentage:5.1f}%)")
        
        # Log total
        self.logger.info("-" * 50)
        total_str = self._format_duration(total_duration)
        self.logger.info(f"{'Total':.<40} {total_str:>10} (100.0%)")
        self.logger.info("-" * 50)
