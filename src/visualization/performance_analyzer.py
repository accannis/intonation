import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import h5py
from datetime import datetime
import os
import json
from pathlib import Path

class PerformanceAnalyzer:
    def __init__(self, save_dir: str = "performance_history"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize performance metrics
        self.metrics = {
            'pitch_accuracy': [],
            'rhythm_stability': [],
            'pronunciation_clarity': [],
            'breath_control': [],
            'overall_consistency': []
        }
        
        # Load existing history if available
        self.history_file = self.save_dir / "performance_history.h5"
        self.load_history()
        
    def load_history(self):
        """Load performance history from file"""
        if self.history_file.exists():
            with h5py.File(self.history_file, 'r') as f:
                for metric in self.metrics:
                    if metric in f:
                        self.metrics[metric] = list(f[metric][()])
                        
    def save_history(self):
        """Save performance history to file"""
        with h5py.File(self.history_file, 'w') as f:
            for metric, values in self.metrics.items():
                if values:  # Only save non-empty metrics
                    f.create_dataset(metric, data=values)
                    
    def calculate_detailed_metrics(self,
                                 audio_features: Dict,
                                 scores: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate detailed performance metrics
        
        Args:
            audio_features: Dictionary of audio features
            scores: Dictionary of current scores
            
        Returns:
            Dictionary of detailed metrics
        """
        # Pitch accuracy (based on melody score)
        pitch_accuracy = scores['melody_score']
        
        # Rhythm stability (based on tempo consistency)
        if 'onset_times' in audio_features:
            onset_intervals = np.diff(audio_features['onset_times'])
            rhythm_stability = 1.0 - np.std(onset_intervals) / np.mean(onset_intervals)
        else:
            rhythm_stability = 0.0
            
        # Pronunciation clarity (based on lyric score)
        pronunciation_clarity = scores['lyric_score']
        
        # Breath control (based on amplitude envelope)
        if 'amplitude_envelope' in audio_features:
            envelope = audio_features['amplitude_envelope']
            breath_control = 1.0 - np.std(envelope) / np.mean(envelope)
        else:
            breath_control = 0.0
            
        # Overall consistency
        overall_consistency = np.mean([
            pitch_accuracy,
            rhythm_stability,
            pronunciation_clarity,
            breath_control
        ])
        
        # Update metrics history
        self.metrics['pitch_accuracy'].append(pitch_accuracy)
        self.metrics['rhythm_stability'].append(rhythm_stability)
        self.metrics['pronunciation_clarity'].append(pronunciation_clarity)
        self.metrics['breath_control'].append(breath_control)
        self.metrics['overall_consistency'].append(overall_consistency)
        
        # Save updated history
        self.save_history()
        
        return {
            'pitch_accuracy': pitch_accuracy,
            'rhythm_stability': rhythm_stability,
            'pronunciation_clarity': pronunciation_clarity,
            'breath_control': breath_control,
            'overall_consistency': overall_consistency
        }
        
    def generate_performance_report(self) -> Tuple[Dict, List[str]]:
        """
        Generate a comprehensive performance report
        
        Returns:
            Tuple of (metrics_summary, insights)
        """
        if not any(self.metrics.values()):
            return {}, ["Not enough data for analysis"]
            
        # Convert metrics to pandas DataFrame for analysis
        df = pd.DataFrame(self.metrics)
        
        # Calculate summary statistics
        summary = {
            'averages': df.mean().to_dict(),
            'improvements': (df.iloc[-1] - df.iloc[0]).to_dict() if len(df) > 1 else {},
            'best_scores': df.max().to_dict(),
            'consistency': (1 - df.std()).to_dict()
        }
        
        # Generate insights
        insights = []
        
        # Overall progress
        if len(df) > 1:
            overall_improvement = summary['improvements']['overall_consistency']
            if overall_improvement > 0.1:
                insights.append("Significant improvement in overall performance! 🌟")
            elif overall_improvement > 0:
                insights.append("Steady progress being made. Keep it up! 📈")
            else:
                insights.append("Focus on maintaining consistency in your practice. 🎯")
                
        # Specific metrics analysis
        for metric in ['pitch_accuracy', 'rhythm_stability', 'pronunciation_clarity', 'breath_control']:
            avg = summary['averages'][metric]
            if avg < 0.5:
                insights.append(f"Work on improving your {metric.replace('_', ' ')}. 💪")
            elif avg > 0.8:
                insights.append(f"Excellent {metric.replace('_', ' ')}! 🌟")
                
        # Save detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'insights': insights
        }
        
        report_file = self.save_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        return summary, insights
        
    def plot_performance_trends(self) -> Tuple[plt.Figure, plt.Figure]:
        """
        Generate performance trend visualizations
        
        Returns:
            Tuple of (metrics_figure, correlation_figure)
        """
        df = pd.DataFrame(self.metrics)
        
        # Create metrics trend plot
        plt.style.use('seaborn')
        metrics_fig, ax1 = plt.subplots(figsize=(12, 6))
        
        for metric in self.metrics:
            ax1.plot(df[metric], label=metric.replace('_', ' ').title())
            
        ax1.set_title('Performance Metrics Over Time')
        ax1.set_xlabel('Session')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True)
        
        # Create correlation heatmap
        correlation_fig, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax2)
        ax2.set_title('Metrics Correlation Heatmap')
        
        # Save plots
        metrics_fig.savefig(self.save_dir / 'performance_trends.png')
        correlation_fig.savefig(self.save_dir / 'metrics_correlation.png')
        
        return metrics_fig, correlation_fig
