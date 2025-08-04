"""
Asynchronous processing capabilities for IoT Edge Anomaly Detection.

This module provides async processing, thread pools, and concurrent data handling
to improve throughput and responsiveness on multi-core edge devices.
"""
import asyncio
import threading
import queue
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from dataclasses import dataclass
import torch
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Represents a data processing task."""
    task_id: str
    data: torch.Tensor
    timestamp: float
    priority: int = 1  # Lower numbers = higher priority
    callback: Optional[Callable] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __lt__(self, other):
        """Make ProcessingTask comparable for priority queue."""
        if not isinstance(other, ProcessingTask):
            return NotImplemented
        return self.priority < other.priority


class AsyncDataProcessor:
    """
    Asynchronous data processor for handling multiple sensor streams concurrently.
    
    Provides thread pool execution, priority queuing, and async processing
    for improved throughput on multi-core edge devices.
    """
    
    def __init__(self, max_workers: int = 4, max_queue_size: int = 100):
        """
        Initialize async processor.
        
        Args:
            max_workers: Maximum number of worker threads
            max_queue_size: Maximum size of processing queue
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        # Thread pool for CPU-bound tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="iot-processor")
        
        # Priority queue for tasks
        self.task_queue = queue.PriorityQueue(maxsize=max_queue_size)
        
        # Processing statistics
        self.processed_count = 0
        self.error_count = 0
        self.queue_full_count = 0
        
        # Worker control
        self.running = False
        self.worker_threads: List[threading.Thread] = []
        
        # Results storage
        self.results_queue = queue.Queue()
        self.pending_tasks: Dict[str, Future] = {}
        
        logger.info(f"Async processor initialized with {max_workers} workers, queue size {max_queue_size}")
    
    def start(self):
        """Start async processing workers."""
        if self.running:
            logger.warning("Async processor already running")
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop, 
                name=f"async-worker-{i}",
                daemon=True
            )
            worker.start()
            self.worker_threads.append(worker)
        
        logger.info(f"Started {len(self.worker_threads)} async processing workers")
    
    def stop(self, timeout: float = 30.0):
        """Stop async processing workers."""
        if not self.running:
            return
        
        logger.info("Stopping async processor...")
        self.running = False
        
        # Add poison pills to wake up workers
        for _ in range(self.max_workers):
            try:
                self.task_queue.put((0, time.time(), None), timeout=1.0)
            except queue.Full:
                pass
        
        # Wait for workers to finish
        start_time = time.time()
        for worker in self.worker_threads:
            remaining_timeout = max(0, timeout - (time.time() - start_time))
            worker.join(timeout=remaining_timeout)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Async processor stopped")
    
    def submit_task(self, task: ProcessingTask) -> Optional[str]:
        """
        Submit task for async processing.
        
        Args:
            task: Processing task to submit
            
        Returns:
            Task ID if submitted successfully, None if queue full
        """
        try:
            # Use tuple with priority and unique timestamp for ordering
            queue_item = (-task.priority, task.timestamp, task)
            self.task_queue.put(queue_item, block=False)
            return task.task_id
        except queue.Full:
            self.queue_full_count += 1
            logger.warning(f"Task queue full, dropping task {task.task_id}")
            return None
    
    def submit_data(self, data: torch.Tensor, processor_func: Callable, 
                   priority: int = 1, metadata: Optional[Dict] = None) -> Future:
        """
        Submit data for async processing.
        
        Args:
            data: Input tensor data
            processor_func: Function to process the data
            priority: Task priority (lower = higher priority)
            metadata: Optional metadata
            
        Returns:
            Future object for the result
        """
        task_id = f"task_{int(time.time() * 1000)}_{id(data)}"
        future = Future()
        
        task = ProcessingTask(
            task_id=task_id,
            data=data,
            timestamp=time.time(),
            priority=priority,
            metadata=metadata or {}
        )
        
        # Store future for this task
        self.pending_tasks[task_id] = (future, processor_func)
        
        # Submit task
        if self.submit_task(task):
            return future
        else:
            # Queue full - return failed future
            future.set_exception(Exception("Processing queue is full"))
            return future
    
    def _worker_loop(self):
        """Main worker loop for processing tasks."""
        worker_name = threading.current_thread().name
        logger.debug(f"Worker {worker_name} started")
        
        while self.running:
            try:
                # Get task from queue with timeout
                queue_item = self.task_queue.get(timeout=1.0)
                
                # Extract task from tuple (priority, timestamp, task)
                if len(queue_item) == 3:
                    priority, timestamp, task = queue_item
                else:
                    # Handle old format or poison pill
                    priority, task = queue_item
                
                # Check for poison pill (stop signal)
                if task is None:
                    break
                
                # Process task
                self._process_task(task)
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                self.error_count += 1
        
        logger.debug(f"Worker {worker_name} stopped")
    
    def _process_task(self, task: ProcessingTask):
        """Process a single task."""
        try:
            # Get future and processor function
            if task.task_id not in self.pending_tasks:
                logger.warning(f"No pending task found for {task.task_id}")
                return
            
            future, processor_func = self.pending_tasks.pop(task.task_id)
            
            # Process data
            start_time = time.time()
            result = processor_func(task.data)
            processing_time = time.time() - start_time
            
            # Set result
            future.set_result({
                'result': result,
                'processing_time': processing_time,
                'task_id': task.task_id,
                'metadata': task.metadata
            })
            
            # Call callback if provided
            if task.callback:
                task.callback(result, task.metadata)
            
            self.processed_count += 1
            
        except Exception as e:
            logger.error(f"Task processing failed for {task.task_id}: {e}")
            self.error_count += 1
            
            # Set exception on future if still pending
            if task.task_id in self.pending_tasks:
                future, _ = self.pending_tasks.pop(task.task_id)
                future.set_exception(e)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "running": self.running,
            "max_workers": self.max_workers,
            "active_workers": len([t for t in self.worker_threads if t.is_alive()]),
            "queue_size": self.task_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "queue_full_count": self.queue_full_count,
            "pending_tasks": len(self.pending_tasks),
            "queue_utilization": self.task_queue.qsize() / self.max_queue_size if self.max_queue_size > 0 else 0
        }


class StreamProcessor:
    """
    Stream processor for handling continuous sensor data streams.
    
    Provides buffering, windowing, and async processing for real-time
    anomaly detection on multiple sensor streams.
    """
    
    def __init__(self, window_size: int = 20, overlap: int = 10, 
                 buffer_size: int = 1000):
        """
        Initialize stream processor.
        
        Args:
            window_size: Size of processing windows
            overlap: Overlap between windows
            buffer_size: Size of data buffers per stream
        """
        self.window_size = window_size
        self.overlap = overlap
        self.buffer_size = buffer_size
        
        # Data buffers for each stream
        self.stream_buffers: Dict[str, deque] = {}
        self.stream_locks: Dict[str, threading.Lock] = {}
        
        # Processing results
        self.results_buffer: deque = deque(maxlen=1000)
        
        # Async processor
        self.async_processor = AsyncDataProcessor(max_workers=2)
        
        logger.info(f"Stream processor initialized (window={window_size}, overlap={overlap})")
    
    def start(self):
        """Start stream processing."""
        self.async_processor.start()
        logger.info("Stream processor started")
    
    def stop(self):
        """Stop stream processing."""
        self.async_processor.stop()
        logger.info("Stream processor stopped")
    
    def add_stream_data(self, stream_id: str, data: torch.Tensor) -> bool:
        """
        Add data to stream buffer.
        
        Args:
            stream_id: Identifier for the data stream
            data: Tensor data point to add
            
        Returns:
            True if data was added, False if buffer full
        """
        # Initialize stream if new
        if stream_id not in self.stream_buffers:
            self.stream_buffers[stream_id] = deque(maxlen=self.buffer_size)
            self.stream_locks[stream_id] = threading.Lock()
        
        # Add data to buffer
        with self.stream_locks[stream_id]:
            if len(self.stream_buffers[stream_id]) >= self.buffer_size:
                logger.warning(f"Stream {stream_id} buffer full, dropping old data")
                return False
            
            self.stream_buffers[stream_id].append({
                'data': data,
                'timestamp': time.time()
            })
        
        # Check if we have enough data for processing
        if len(self.stream_buffers[stream_id]) >= self.window_size:
            self._try_process_window(stream_id)
        
        return True
    
    def _try_process_window(self, stream_id: str):
        """Try to process a window of data from stream."""
        with self.stream_locks[stream_id]:
            buffer = self.stream_buffers[stream_id]
            
            if len(buffer) < self.window_size:
                return
            
            # Extract window data
            window_data = []
            window_timestamps = []
            
            for i in range(-self.window_size, 0):  # Get last window_size items
                item = buffer[i]
                window_data.append(item['data'])
                window_timestamps.append(item['timestamp'])
            
            # Stack into tensor
            try:
                window_tensor = torch.stack(window_data, dim=0)
                window_tensor = window_tensor.unsqueeze(0)  # Add batch dimension
                
            except Exception as e:
                logger.error(f"Failed to create window tensor for {stream_id}: {e}")
                return
        
        # Submit for async processing
        def processor_func(data: torch.Tensor):
            # This would call your anomaly detection model
            # For now, return dummy result
            return {
                'anomaly_score': torch.mean(data).item(),
                'is_anomaly': torch.mean(data).item() > 0.5,
                'window_start': min(window_timestamps),
                'window_end': max(window_timestamps)
            }
        
        future = self.async_processor.submit_data(
            window_tensor, 
            processor_func,
            metadata={'stream_id': stream_id, 'window_timestamps': window_timestamps}
        )
        
        # Add callback to store results
        def result_callback(future_obj):
            try:
                result = future_obj.result(timeout=1.0)
                self.results_buffer.append({
                    'stream_id': stream_id,
                    'timestamp': time.time(),
                    'result': result
                })
            except Exception as e:
                logger.error(f"Failed to get result for {stream_id}: {e}")
        
        # Set callback
        future.add_done_callback(result_callback)
    
    def get_recent_results(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent processing results."""
        return list(self.results_buffer)[-count:]
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get stream processing statistics."""
        stats = {
            'streams': {},
            'total_streams': len(self.stream_buffers),
            'async_processor': self.async_processor.get_stats(),
            'results_buffer_size': len(self.results_buffer)
        }
        
        # Add per-stream stats
        for stream_id, buffer in self.stream_buffers.items():
            with self.stream_locks[stream_id]:
                stats['streams'][stream_id] = {
                    'buffer_size': len(buffer),
                    'buffer_utilization': len(buffer) / self.buffer_size,
                    'latest_timestamp': buffer[-1]['timestamp'] if buffer else None
                }
        
        return stats


# Global instances
async_processor = AsyncDataProcessor()
stream_processor = StreamProcessor()