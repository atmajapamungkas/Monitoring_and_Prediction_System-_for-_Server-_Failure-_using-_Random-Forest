import tkinter as tk
from tkinter import ttk
import psutil
from ping3 import ping
import pandas as pd
import numpy as np
from collections import deque
from sklearn.ensemble import RandomForestRegressor  # Berubah ke Regressor untuk prediksi waktu
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=UserWarning)

class ServerHealthMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Server Failure Prediction System")
        self.root.geometry("1400x900")
        
        # Load dataset and train model
        self.load_dataset()
        
        # Initialize data storage
        self.initialize_deques()
        
        # Setup GUI
        self.setup_gui()
        
        # Start monitoring
        self.start_monitoring()

    def load_dataset(self):
        """Load dataset and train time-to-failure prediction model"""
        try:
            # Dataset harus mengandung kolom 'time_to_failure' dalam satuan detik
            self.df = pd.read_csv('time_to_failure_dataset.csv')
            
            # Preprocessing
            if 'timestamp' in self.df.columns:
                self.df = self.df.drop(columns=['timestamp'])
            
            # Pastikan ada kolom target 'time_to_failure'
            if 'time_to_failure' not in self.df.columns:
                raise ValueError("Dataset must contain 'time_to_failure' column")
            
            # Split features and target
            self.X = self.df.drop('time_to_failure', axis=1)
            self.y = self.df['time_to_failure']
            
            # Train regression model
            self.model = RandomForestRegressor(n_estimators=100)
            self.model.fit(self.X, self.y)
            
            print("Model trained successfully")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def initialize_deques(self):
        """Initialize data storage"""
        self.cpu_data = deque(maxlen=100)
        self.memory_data = deque(maxlen=100)
        self.disk_data = deque(maxlen=100)
        self.latency_data = deque(maxlen=100)
        self.ttf_data = deque(maxlen=100)  # Time-to-failure data
        self.predicted_failure_time = None

    def setup_gui(self):
        """Setup GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status panel
        status_frame = ttk.LabelFrame(main_frame, text="Server Status")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(
            status_frame, 
            text="Status: INITIALIZING...", 
            font=("Arial", 12)
        )
        self.status_label.pack(pady=5)
        
        # Time-to-failure prediction
        self.ttf_label = ttk.Label(
            status_frame,
            text="Predicted Time to Failure: --",
            font=("Arial", 12)
        )
        self.ttf_label.pack(pady=5)
        
        # Countdown display
        self.countdown_label = ttk.Label(
            status_frame,
            text="Countdown: --:--:--",
            font=("Arial", 14, "bold")
        )
        self.countdown_label.pack(pady=5)
        
        # Metrics frame
        metrics_frame = ttk.Frame(main_frame)
        metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Current metrics labels
        ttk.Label(metrics_frame, text="Current Metrics:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w")
        
        metrics = ["CPU Usage:", "Memory Usage:", "Disk Usage:", "Latency:"]
        self.metric_labels = []
        
        for i, metric in enumerate(metrics):
            ttk.Label(metrics_frame, text=metric).grid(row=i+1, column=0, sticky="w")
            label = ttk.Label(metrics_frame, text="0%")
            label.grid(row=i+1, column=1, sticky="w")
            self.metric_labels.append(label)
        
        # Setup plots
        self.setup_plots(main_frame)

    def setup_plots(self, parent):
        """Setup matplotlib plots"""
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4), (self.ax5, self.ax6)) = plt.subplots(3, 2, figsize=(12, 10))
        self.fig.tight_layout(pad=4.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def start_monitoring(self):
        """Start the monitoring process"""
        self.update_metrics()

    def get_system_metrics(self):
        """Collect current system metrics"""
        return {
            'cpu': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory().percent,
            'disk': self.get_disk_usage(),
            'latency': self.get_latency()
        }

    def get_latency(self):
        """Get network latency in ms"""
        try:
            latency = ping('192.168.1.1') or 0
            return latency * 1000  # Convert to milliseconds
        except:
            return float('inf')

    def get_disk_usage(self):
        """Get disk usage percentage"""
        try:
            for drive in ['C:\\', 'D:\\', '/']:
                try:
                    return psutil.disk_usage(drive).percent
                except:
                    continue
            return 0
        except Exception as e:
            print(f"Error accessing disk: {e}")
            return 0

    def predict_time_to_failure(self, metrics):
        """Predict remaining time until failure (in seconds)"""
        try:
            # Create sample with same features as training data
            sample = pd.DataFrame([[
                metrics['cpu'],
                metrics['memory'],
                metrics['disk'],
                metrics['latency'],
                0, 0, 0  # Placeholder for other features
            ]], columns=self.X.columns)
            
            # Handle NaN/inf values
            sample = np.nan_to_num(sample, nan=0.0, posinf=1000)
            
            # Predict time to failure (in seconds)
            ttf = self.model.predict(sample)[0]
            return max(0, ttf)  # Ensure non-negative
        except Exception as e:
            print(f"Prediction error: {e}")
            return float('inf')  # Return infinity if prediction fails

    def update_gui(self, metrics, ttf):
        """Update all GUI elements"""
        # Update metric labels
        self.metric_labels[0].config(text=f"{metrics['cpu']:.1f}%")
        self.metric_labels[1].config(text=f"{metrics['memory']:.1f}%")
        self.metric_labels[2].config(text=f"{metrics['disk']:.1f}%")
        self.metric_labels[3].config(text=f"{metrics['latency']:.1f} ms")
        
        # Determine status
        if ttf <= 0:
            status_text = "FAILURE IMMINENT!"
            status_color = "red"
        elif ttf < 300:  # 5 minutes
            status_text = "CRITICAL"
            status_color = "red"
        elif ttf < 1800:  # 30 minutes
            status_text = "WARNING"
            status_color = "orange"
        elif ttf < 3600:  # 1 hour
            status_text = "STABLE BUT WATCH"
            status_color = "yellow"
        else:
            status_text = "HEALTHY"
            status_color = "green"
        
        self.status_label.config(text=f"Status: {status_text}", foreground=status_color)
        
        # Update time-to-failure prediction
        if ttf == float('inf'):
            self.ttf_label.config(text="Predicted Time to Failure: UNKNOWN")
            self.countdown_label.config(text="Countdown: --:--:--")
        else:
            # Format as hours:minutes:seconds
            hours, remainder = divmod(ttf, 3600)
            minutes, seconds = divmod(remainder, 60)
            ttf_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            
            self.ttf_label.config(text=f"Predicted Time to Failure: {ttf_str}")
            
            # Update countdown
            if self.predicted_failure_time is None or ttf < 3600:  # Update more frequently when close to failure
                self.predicted_failure_time = datetime.now() + timedelta(seconds=ttf)
            
            remaining = self.predicted_failure_time - datetime.now()
            remaining_seconds = max(0, remaining.total_seconds())
            
            # Format countdown
            hours, remainder = divmod(remaining_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            countdown_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            
            self.countdown_label.config(text=f"Countdown: {countdown_str}")

    def update_plots(self, metrics, ttf):
        """Update all plots"""
        try:
            # Clear and update each plot
            self.ax1.clear()
            self.ax1.plot(list(self.cpu_data), label='CPU Usage', color='blue')
            self.ax1.set_ylim(0, 100)
            self.ax1.set_title("CPU Usage (%)")
            self.ax1.legend(loc='upper right')
            
            self.ax2.clear()
            self.ax2.plot(list(self.memory_data), label='Memory Usage', color='green')
            self.ax2.set_ylim(0, 100)
            self.ax2.set_title("Memory Usage (%)")
            self.ax2.legend(loc='upper right')
            
            self.ax3.clear()
            self.ax3.plot(list(self.disk_data), label='Disk Usage', color='red')
            self.ax3.set_ylim(0, 100)
            self.ax3.set_title("Disk Usage (%)")
            self.ax3.legend(loc='upper right')
            
            self.ax4.clear()
            self.ax4.plot(list(self.latency_data), label='Latency', color='purple')
            self.ax4.set_title("Network Latency (ms)")
            self.ax4.legend(loc='upper right')
            
            self.ax5.clear()
            self.ax5.plot(list(self.ttf_data), label='Time to Failure', color='orange')
            self.ax5.set_title("Predicted Time to Failure (sec)")
            self.ax5.legend(loc='upper right')
            
            # Health status indicator
            self.ax6.clear()
            health_status = min(100, max(0, (ttf / 3600) * 100)) if ttf != float('inf') else 100
            self.ax6.bar(['Health Status'], [health_status], color=self.get_health_color(ttf))
            self.ax6.set_ylim(0, 100)
            self.ax6.set_title("Server Health Status")
            
            self.canvas.draw()
        except Exception as e:
            print(f"Plot update error: {e}")

    def get_health_color(self, ttf):
        """Get color based on time to failure"""
        if ttf <= 0:
            return 'red'
        elif ttf < 300:  # 5 minutes
            return 'darkred'
        elif ttf < 1800:  # 30 minutes
            return 'orange'
        elif ttf < 3600:  # 1 hour
            return 'yellow'
        else:
            return 'green'

    def update_metrics(self):
        """Main update function"""
        try:
            # Get current metrics
            metrics = self.get_system_metrics()
            
            # Store metrics in deques
            self.cpu_data.append(metrics['cpu'])
            self.memory_data.append(metrics['memory'])
            self.disk_data.append(metrics['disk'])
            self.latency_data.append(metrics['latency'])
            
            # Predict time to failure
            ttf = self.predict_time_to_failure(metrics)
            self.ttf_data.append(ttf if ttf != float('inf') else 3600)  # Cap at 1 hour for plot
            
            # Update GUI
            self.update_gui(metrics, ttf)
            self.update_plots(metrics, ttf)
            
        except Exception as e:
            print(f"Update error: {e}")
            self.status_label.config(text="Status: UPDATE ERROR", foreground="red")
        
        # Schedule next update
        self.root.after(1000, self.update_metrics)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ServerHealthMonitor(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
