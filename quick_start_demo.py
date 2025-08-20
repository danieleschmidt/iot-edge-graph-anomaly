#!/usr/bin/env python3
"""
🚀 TERRAGON IoT Anomaly Detection - One-Command Quick Start Demo

This script provides an instant demonstration of all 5 breakthrough AI algorithms
with zero configuration required. Perfect for researchers, developers, and 
decision-makers who want to see the system's capabilities immediately.

Usage:
    python quick_start_demo.py
    
    # Or with options:
    python quick_start_demo.py --samples 100 --visualize --save-results
"""

import os
import sys
import time
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    import torch
    from iot_edge_anomaly.models.advanced_hybrid_integration import create_advanced_hybrid_system
    print("✅ PyTorch and models loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Installing dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    import torch
    from iot_edge_anomaly.models.advanced_hybrid_integration import create_advanced_hybrid_system


class QuickStartDemo:
    """Ultra-fast demonstration of the complete anomaly detection system."""
    
    def __init__(self, num_samples: int = 50, visualize: bool = True):
        self.num_samples = num_samples
        self.visualize = visualize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 Initializing on {self.device}")
        print(f"📊 Will process {num_samples} samples")
    
    def generate_realistic_data(self):
        """Generate realistic IoT sensor data with known anomalies."""
        print("🎯 Generating realistic industrial sensor data...")
        
        # Simulate 5 industrial sensors: temp, pressure, flow, vibration, power
        data = []
        labels = []
        
        for i in range(self.num_samples):
            # Normal operation (80% of samples)
            if i < int(self.num_samples * 0.8):
                # Normal patterns with realistic sensor correlations
                t = np.linspace(0, 4*np.pi, 20)
                temp = 25 + 10 * np.sin(t) + np.random.normal(0, 1, 20)
                pressure = 2 + 0.5 * np.cos(t) + np.random.normal(0, 0.1, 20)
                flow = 1.5 + 0.3 * np.sin(t + np.pi/4) + np.random.normal(0, 0.05, 20)
                vibration = 0.1 + 0.05 * np.random.random(20)
                power = 500 + 100 * (temp - 25) / 10 + np.random.normal(0, 20, 20)
                labels.append(0)  # Normal
            else:
                # Anomalous patterns (20% of samples)
                t = np.linspace(0, 4*np.pi, 20)
                
                # Different types of anomalies
                anomaly_type = (i - int(self.num_samples * 0.8)) % 4
                
                if anomaly_type == 0:  # Temperature spike
                    temp = 25 + 10 * np.sin(t) + 15 + np.random.normal(0, 2, 20)
                    pressure = 2 + 0.5 * np.cos(t) + np.random.normal(0, 0.1, 20)
                    flow = 1.5 + 0.3 * np.sin(t + np.pi/4) + np.random.normal(0, 0.05, 20)
                    vibration = 0.1 + 0.15 * np.random.random(20)  # Increased vibration
                    power = 500 + 100 * (temp - 25) / 10 + 200 + np.random.normal(0, 50, 20)
                    
                elif anomaly_type == 1:  # Pressure drop
                    temp = 25 + 10 * np.sin(t) + np.random.normal(0, 1, 20)
                    pressure = 0.5 + 0.2 * np.cos(t) + np.random.normal(0, 0.1, 20)  # Low pressure
                    flow = 0.8 + 0.1 * np.sin(t + np.pi/4) + np.random.normal(0, 0.05, 20)  # Reduced flow
                    vibration = 0.1 + 0.05 * np.random.random(20)
                    power = 500 + 100 * (temp - 25) / 10 + np.random.normal(0, 20, 20)
                    
                elif anomaly_type == 2:  # Oscillation
                    temp = 25 + 10 * np.sin(t) + 5 * np.sin(10*t) + np.random.normal(0, 1, 20)
                    pressure = 2 + 0.5 * np.cos(t) + 0.3 * np.sin(15*t) + np.random.normal(0, 0.1, 20)
                    flow = 1.5 + 0.3 * np.sin(t + np.pi/4) + np.random.normal(0, 0.05, 20)
                    vibration = 0.1 + 0.2 * np.sin(20*t) + 0.05 * np.random.random(20)
                    power = 500 + 100 * (temp - 25) / 10 + np.random.normal(0, 20, 20)
                    
                else:  # Sensor drift
                    drift = np.linspace(0, 10, 20)
                    temp = 25 + 10 * np.sin(t) + drift + np.random.normal(0, 1, 20)
                    pressure = 2 + 0.5 * np.cos(t) + np.random.normal(0, 0.1, 20)
                    flow = 1.5 + 0.3 * np.sin(t + np.pi/4) + np.random.normal(0, 0.05, 20)
                    vibration = 0.1 + 0.05 * np.random.random(20)
                    power = 500 + 100 * (temp - 25) / 10 + np.random.normal(0, 20, 20)
                
                labels.append(1)  # Anomaly
            
            # Stack sensors into single sample
            sample = np.stack([temp, pressure, flow, vibration, power], axis=1)
            data.append(sample)
        
        # Convert to tensors
        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        print(f"   ✅ Generated {len(data)} samples ({sum(labels)} anomalies)")
        return data_tensor, labels_tensor
    
    def run_demonstration(self):
        """Run the complete demonstration."""
        print("\n" + "="*60)
        print("🚀 TERRAGON IoT ANOMALY DETECTION QUICK DEMO")
        print("="*60)
        
        start_time = time.time()
        
        # 1. Generate data
        data, labels = self.generate_realistic_data()
        
        # 2. Initialize all 5 advanced models
        print("\n🧠 Initializing 5 breakthrough AI algorithms...")
        ensemble_config = {
            'enable_transformer_vae': True,
            'enable_sparse_gat': True, 
            'enable_physics_informed': True,
            'enable_self_supervised': True,
            'enable_quantum_classical': True,
            'ensemble_method': 'dynamic_weighting',
            'uncertainty_quantification': True
        }
        
        try:
            ensemble = create_advanced_hybrid_system(ensemble_config)
            print("   ✅ All models initialized successfully")
        except Exception as e:
            print(f"   ⚠️  Using simplified ensemble due to: {e}")
            # Fallback to basic ensemble
            ensemble_config = {'ensemble_method': 'simple_average'}
            ensemble = create_advanced_hybrid_system(ensemble_config)
        
        # 3. Run predictions
        print("\n⚡ Running anomaly detection on all samples...")
        
        predictions = []
        anomaly_scores = []
        inference_times = []
        
        with torch.no_grad():
            for i, sample in enumerate(data):
                sample_start = time.time()
                
                try:
                    result = ensemble.predict(sample.unsqueeze(0))
                    if isinstance(result, dict):
                        score = result.get('anomaly_score', 0.5)
                        pred = 1 if score > 0.5 else 0
                    else:
                        score = float(result)
                        pred = 1 if score > 0.5 else 0
                except Exception as e:
                    print(f"   ⚠️  Sample {i} failed: {e}")
                    score = 0.5
                    pred = 0
                
                inference_time = time.time() - sample_start
                
                predictions.append(pred)
                anomaly_scores.append(score)
                inference_times.append(inference_time)
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i+1}/{len(data)} samples...")
        
        # 4. Calculate metrics
        predictions = np.array(predictions)
        labels_np = labels.numpy()
        
        # Basic metrics
        accuracy = np.mean(predictions == labels_np)
        precision = np.sum((predictions == 1) & (labels_np == 1)) / max(np.sum(predictions == 1), 1)
        recall = np.sum((predictions == 1) & (labels_np == 1)) / max(np.sum(labels_np == 1), 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 1e-6)
        
        avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
        total_time = time.time() - start_time
        
        # 5. Display results
        print("\n" + "="*60)
        print("📊 DEMONSTRATION RESULTS")
        print("="*60)
        
        print(f"\n🎯 PERFORMANCE METRICS:")
        print(f"   Accuracy:      {accuracy:.1%}")
        print(f"   Precision:     {precision:.1%}")
        print(f"   Recall:        {recall:.1%}")
        print(f"   F1-Score:      {f1_score:.1%}")
        
        print(f"\n⚡ SPEED METRICS:")
        print(f"   Avg Inference: {avg_inference_time:.2f} ms")
        print(f"   Total Time:    {total_time:.2f} seconds")
        print(f"   Throughput:    {len(data)/total_time:.1f} samples/sec")
        
        print(f"\n🔍 DETECTION SUMMARY:")
        print(f"   Total Samples:     {len(data)}")
        print(f"   True Anomalies:    {sum(labels_np)}")
        print(f"   Detected:          {sum(predictions)}")
        print(f"   True Positives:    {sum((predictions == 1) & (labels_np == 1))}")
        print(f"   False Positives:   {sum((predictions == 1) & (labels_np == 0))}")
        print(f"   False Negatives:   {sum((predictions == 0) & (labels_np == 1))}")
        
        # 6. Visualization
        if self.visualize:
            self.create_visualizations(data, labels_np, predictions, anomaly_scores)
        
        # 7. Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(data),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'avg_inference_time_ms': float(avg_inference_time),
            'total_time_seconds': float(total_time),
            'throughput_samples_per_sec': float(len(data)/total_time),
            'model_config': ensemble_config,
            'device': str(self.device)
        }
        
        print(f"\n💾 SAVING RESULTS...")
        with open('quick_demo_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   ✅ Results saved to: quick_demo_results.json")
        
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print(f"   The Terragon IoT Anomaly Detection System successfully")
        print(f"   processed {len(data)} samples with {f1_score:.1%} F1-score")
        print(f"   in just {total_time:.1f} seconds!")
        
        return results
    
    def create_visualizations(self, data, labels, predictions, scores):
        """Create visualization of results."""
        print(f"\n📈 Creating visualizations...")
        
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Terragon IoT Anomaly Detection - Quick Demo Results', fontsize=16, fontweight='bold')
        
        # 1. Sample sensor data
        sample_idx = 5  # Show a normal sample
        axes[0, 0].plot(data[sample_idx, :, :].numpy())
        axes[0, 0].set_title('Sample Industrial Sensor Data')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Sensor Values')
        axes[0, 0].legend(['Temperature', 'Pressure', 'Flow', 'Vibration', 'Power'])
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Anomaly score distribution
        normal_scores = [scores[i] for i in range(len(scores)) if labels[i] == 0]
        anomaly_scores = [scores[i] for i in range(len(scores)) if labels[i] == 1]
        
        axes[0, 1].hist(normal_scores, bins=20, alpha=0.7, label='Normal', color='green')
        axes[0, 1].hist(anomaly_scores, bins=20, alpha=0.7, label='Anomaly', color='red')
        axes[0, 1].axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        axes[0, 1].set_title('Anomaly Score Distribution')
        axes[0, 1].set_xlabel('Anomaly Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confusion matrix
        tp = sum((predictions == 1) & (labels == 1))
        fp = sum((predictions == 1) & (labels == 0))
        tn = sum((predictions == 0) & (labels == 0))
        fn = sum((predictions == 0) & (labels == 1))
        
        confusion_matrix = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                   yticklabels=['Actual Normal', 'Actual Anomaly'],
                   ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        
        # 4. Timeline view
        x = range(len(scores))
        colors = ['red' if labels[i] == 1 else 'green' for i in range(len(labels))]
        
        axes[1, 1].scatter(x, scores, c=colors, alpha=0.6)
        axes[1, 1].axhline(y=0.5, color='black', linestyle='--', label='Threshold')
        axes[1, 1].set_title('Anomaly Detection Timeline')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Anomaly Score')
        axes[1, 1].legend(['Threshold', 'Normal (True)', 'Anomaly (True)'])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quick_demo_visualization.png', dpi=300, bbox_inches='tight')
        print(f"   ✅ Visualization saved to: quick_demo_visualization.png")
        
        # Show plot if possible
        try:
            plt.show()
        except:
            print("   📱 (Display not available - saved to file)")


def main():
    """Main entry point for quick start demo."""
    parser = argparse.ArgumentParser(description='Terragon IoT Anomaly Detection Quick Demo')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples to generate')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--save-results', action='store_true', help='Save detailed results')
    
    args = parser.parse_args()
    
    print("🚀 Starting Terragon IoT Anomaly Detection Quick Demo...")
    print(f"   Version: 4.0.0 (Autonomous SDLC)")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        demo = QuickStartDemo(
            num_samples=args.samples,
            visualize=args.visualize
        )
        
        results = demo.run_demonstration()
        
        if args.save_results:
            print(f"\n💾 Detailed results available in quick_demo_results.json")
        
        print(f"\n🌟 Thank you for trying Terragon's IoT Anomaly Detection System!")
        print(f"   For production deployment: python -m iot_edge_anomaly.main")
        print(f"   For advanced features: python examples/comprehensive_demo.py")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print(f"   Please check dependencies: pip install -e .")
        return 1


if __name__ == "__main__":
    exit(main())