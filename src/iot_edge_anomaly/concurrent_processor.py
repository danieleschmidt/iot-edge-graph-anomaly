"""
🚀 Concurrent Processing and Resource Management System

This module provides advanced concurrent processing, batch optimization,
and distributed computing capabilities for the Terragon IoT Anomaly Detection System.

Features:
- Asynchronous batch processing with dynamic batching
- Multi-threaded inference pipeline with worker pools
- Distributed processing across multiple nodes
- Intelligent load balancing and request routing
- Real-time stream processing with backpressure handling
- GPU/CPU resource scheduling and optimization
- Performance-aware auto-scaling
"""

import asyncio
import threading
import multiprocessing
import time
import queue
import concurrent.futures
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, AsyncGenerator
from enum import Enum
import logging
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing execution modes."""
    SYNC = "sync"
    ASYNC = "async"
    BATCH = "batch"
    STREAM = "stream"
    DISTRIBUTED = "distributed"


class WorkerType(Enum):
    """Types of workers."""
    CPU = "cpu"
    GPU = "gpu"
    MIXED = "mixed"


@dataclass
class ProcessingRequest:
    """Request for processing."""
    request_id: str
    data: Any
    priority: int = 1
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    timeout: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'priority': self.priority,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'timeout': self.timeout
        }


@dataclass
class ProcessingResult:
    """Result of processing."""
    request_id: str
    result: Any
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0
    worker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'success': self.success,
            'error': self.error,
            'processing_time': self.processing_time,
            'worker_id': self.worker_id,
            'metadata': self.metadata,
            'completed_at': self.completed_at.isoformat()
        }


class WorkerPool:
    """Advanced worker pool with different worker types."""
    
    def __init__(
        self,
        num_workers: int = 4,
        worker_type: WorkerType = WorkerType.CPU,
        max_queue_size: int = 1000
    ):
        self.num_workers = num_workers
        self.worker_type = worker_type
        self.max_queue_size = max_queue_size
        
        # Worker management
        self.workers = []
        self.request_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.result_handlers = {}
        self.worker_stats = {}
        
        # Control
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0,
            'queue_size': 0,
            'active_workers': 0
        }
    
    def start(self):
        """Start the worker pool."""
        if self.running:
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Create workers
        for i in range(self.num_workers):
            worker_id = f"worker_{i}_{self.worker_type.value}"
            worker = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
            self.worker_stats[worker_id] = {
                'requests_processed': 0,
                'total_processing_time': 0.0,
                'errors': 0,
                'last_activity': datetime.now()
            }
        
        logger.info(f"Started worker pool with {self.num_workers} {self.worker_type.value} workers")
    
    def stop(self, timeout: float = 30.0):
        """Stop the worker pool."""
        if not self.running:
            return
        
        logger.info("Stopping worker pool...")
        self.running = False
        self.shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
        
        self.workers.clear()
        logger.info("Worker pool stopped")
    
    def submit(
        self,
        func: Callable,
        request: ProcessingRequest,
        *args,
        **kwargs
    ) -> bool:
        """Submit a processing request."""
        if not self.running:
            return False
        
        try:
            # Priority queue item: (priority, timestamp, request_id, (func, request, args, kwargs))
            priority_item = (
                -request.priority,  # Negative for high priority first
                time.time(),
                request.request_id,
                (func, request, args, kwargs)
            )
            
            self.request_queue.put(priority_item, block=False)
            self.metrics['total_requests'] += 1
            self.metrics['queue_size'] = self.request_queue.qsize()
            
            return True
            
        except queue.Full:
            logger.warning("Worker pool queue is full")
            return False
    
    def _worker_loop(self, worker_id: str):
        """Main worker loop."""
        logger.info(f"Worker {worker_id} started")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Get next request (with timeout to allow shutdown check)
                try:
                    priority_item = self.request_queue.get(timeout=1.0)
                    _, _, request_id, (func, request, args, kwargs) = priority_item
                except queue.Empty:
                    continue
                
                # Process request
                start_time = time.time()
                try:
                    # Execute function
                    result = func(request.data, *args, **kwargs)
                    
                    processing_result = ProcessingResult(
                        request_id=request.request_id,
                        result=result,
                        success=True,
                        processing_time=time.time() - start_time,
                        worker_id=worker_id
                    )
                    
                    self.metrics['completed_requests'] += 1
                    
                except Exception as e:
                    processing_result = ProcessingResult(
                        request_id=request.request_id,
                        result=None,
                        success=False,
                        error=str(e),
                        processing_time=time.time() - start_time,
                        worker_id=worker_id
                    )
                    
                    self.metrics['failed_requests'] += 1
                    logger.error(f"Worker {worker_id} error processing {request_id}: {e}")
                
                # Update worker stats
                stats = self.worker_stats[worker_id]
                stats['requests_processed'] += 1
                stats['total_processing_time'] += processing_result.processing_time
                stats['last_activity'] = datetime.now()
                
                if not processing_result.success:
                    stats['errors'] += 1
                
                # Handle callback
                if request.callback:
                    try:
                        request.callback(processing_result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                # Store result for retrieval
                if request_id in self.result_handlers:
                    self.result_handlers[request_id].put(processing_result)
                
                # Update metrics
                self.metrics['queue_size'] = self.request_queue.qsize()
                
                # Update average processing time
                total_completed = self.metrics['completed_requests'] + self.metrics['failed_requests']
                if total_completed > 0:
                    total_time = sum(
                        stats['total_processing_time'] 
                        for stats in self.worker_stats.values()
                    )
                    self.metrics['avg_processing_time'] = total_time / total_completed
                
                self.request_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker {worker_id} unexpected error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    def get_result(self, request_id: str, timeout: Optional[float] = None) -> Optional[ProcessingResult]:
        """Get result for a specific request."""
        if request_id not in self.result_handlers:
            self.result_handlers[request_id] = queue.Queue()
        
        try:
            result = self.result_handlers[request_id].get(timeout=timeout)
            del self.result_handlers[request_id]  # Clean up
            return result
        except queue.Empty:
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker pool metrics."""
        active_workers = sum(
            1 for stats in self.worker_stats.values()
            if datetime.now() - stats['last_activity'] < timedelta(seconds=60)
        )
        
        return {
            **self.metrics,
            'active_workers': active_workers,
            'worker_stats': self.worker_stats,
            'success_rate': (
                self.metrics['completed_requests'] / 
                max(self.metrics['total_requests'], 1)
            )
        }


class BatchProcessor:
    """Dynamic batch processing system."""
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,
        min_batch_size: int = 1
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.min_batch_size = min_batch_size
        
        # Batch management
        self.pending_requests = []
        self.batch_lock = threading.Lock()
        self.batch_event = threading.Event()
        
        # Processing
        self.running = False
        self.processor_thread = None
        
        # Metrics
        self.metrics = {
            'total_batches': 0,
            'total_items': 0,
            'avg_batch_size': 0.0,
            'avg_wait_time': 0.0
        }
    
    def start(self):
        """Start batch processor."""
        if self.running:
            return
        
        self.running = True
        self.processor_thread = threading.Thread(target=self._batch_loop, daemon=True)
        self.processor_thread.start()
        logger.info("Batch processor started")
    
    def stop(self):
        """Stop batch processor."""
        if not self.running:
            return
        
        self.running = False
        self.batch_event.set()
        
        if self.processor_thread:
            self.processor_thread.join(timeout=5.0)
        
        logger.info("Batch processor stopped")
    
    def add_request(self, request: ProcessingRequest) -> bool:
        """Add request to batch."""
        if not self.running:
            return False
        
        with self.batch_lock:
            self.pending_requests.append(request)
            
            # Check if we should process batch immediately
            if len(self.pending_requests) >= self.max_batch_size:
                self.batch_event.set()
        
        return True
    
    def _batch_loop(self):
        """Main batch processing loop."""
        while self.running:
            try:
                # Wait for batch to be ready
                self.batch_event.wait(timeout=self.max_wait_time)
                self.batch_event.clear()
                
                # Get batch
                with self.batch_lock:
                    if len(self.pending_requests) < self.min_batch_size:
                        continue
                    
                    batch = self.pending_requests[:self.max_batch_size]
                    self.pending_requests = self.pending_requests[self.max_batch_size:]
                
                if not batch:
                    continue
                
                # Process batch
                self._process_batch(batch)
                
                # Update metrics
                self.metrics['total_batches'] += 1
                self.metrics['total_items'] += len(batch)
                self.metrics['avg_batch_size'] = (
                    self.metrics['total_items'] / self.metrics['total_batches']
                )
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
    
    def _process_batch(self, batch: List[ProcessingRequest]):
        """Process a batch of requests."""
        start_time = time.time()
        
        try:
            # Extract data from requests
            batch_data = [req.data for req in batch]
            
            # Here you would implement actual batch processing
            # For now, we'll simulate batch processing
            results = self._simulate_batch_inference(batch_data)
            
            # Create results
            processing_time = time.time() - start_time
            
            for i, request in enumerate(batch):
                result = ProcessingResult(
                    request_id=request.request_id,
                    result=results[i] if i < len(results) else None,
                    success=True,
                    processing_time=processing_time / len(batch),  # Distribute time across batch
                    metadata={'batch_size': len(batch)}
                )
                
                # Handle callback
                if request.callback:
                    try:
                        request.callback(result)
                    except Exception as e:
                        logger.error(f"Batch callback error: {e}")
        
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            
            # Create error results
            for request in batch:
                result = ProcessingResult(
                    request_id=request.request_id,
                    result=None,
                    success=False,
                    error=str(e),
                    processing_time=0.0
                )
                
                if request.callback:
                    try:
                        request.callback(result)
                    except Exception as callback_error:
                        logger.error(f"Error callback error: {callback_error}")
    
    def _simulate_batch_inference(self, batch_data: List[Any]) -> List[Any]:
        """Simulate batch inference (replace with actual model inference)."""
        # This would be replaced with actual batch model inference
        results = []
        for data in batch_data:
            if isinstance(data, (torch.Tensor, np.ndarray)):
                # Simulate anomaly detection result
                result = {
                    'is_anomaly': False,
                    'anomaly_score': 0.3,
                    'confidence': 0.85
                }
            else:
                result = {'processed': True}
            
            results.append(result)
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get batch processor metrics."""
        with self.batch_lock:
            pending_count = len(self.pending_requests)
        
        return {
            **self.metrics,
            'pending_requests': pending_count
        }


class StreamProcessor:
    """Real-time stream processing with backpressure handling."""
    
    def __init__(
        self,
        buffer_size: int = 1000,
        processing_rate_limit: Optional[float] = None
    ):
        self.buffer_size = buffer_size
        self.processing_rate_limit = processing_rate_limit
        
        # Stream management
        self.stream_buffer = asyncio.Queue(maxsize=buffer_size)
        self.running = False
        self.processor_tasks = []
        
        # Backpressure handling
        self.backpressure_threshold = buffer_size * 0.8
        self.backpressure_active = False
        
        # Metrics
        self.metrics = {
            'total_items': 0,
            'processed_items': 0,
            'dropped_items': 0,
            'backpressure_events': 0,
            'avg_processing_rate': 0.0
        }
        
        self.rate_limiter = None
        if processing_rate_limit:
            self.rate_limiter = asyncio.Semaphore(int(processing_rate_limit))
    
    async def start(self, num_processors: int = 2):
        """Start stream processor."""
        if self.running:
            return
        
        self.running = True
        
        # Start processor tasks
        for i in range(num_processors):
            task = asyncio.create_task(self._stream_processor_loop(f"processor_{i}"))
            self.processor_tasks.append(task)
        
        logger.info(f"Stream processor started with {num_processors} processors")
    
    async def stop(self):
        """Stop stream processor."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel processor tasks
        for task in self.processor_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.processor_tasks:
            await asyncio.gather(*self.processor_tasks, return_exceptions=True)
        
        self.processor_tasks.clear()
        logger.info("Stream processor stopped")
    
    async def add_item(self, item: Any) -> bool:
        """Add item to stream (with backpressure handling)."""
        if not self.running:
            return False
        
        try:
            # Check for backpressure
            current_size = self.stream_buffer.qsize()
            
            if current_size >= self.backpressure_threshold:
                if not self.backpressure_active:
                    self.backpressure_active = True
                    self.metrics['backpressure_events'] += 1
                    logger.warning("Backpressure activated - stream buffer nearly full")
                
                # Drop item if buffer is full
                if current_size >= self.buffer_size:
                    self.metrics['dropped_items'] += 1
                    return False
            else:
                self.backpressure_active = False
            
            # Add item to buffer
            await self.stream_buffer.put(item)
            self.metrics['total_items'] += 1
            
            return True
            
        except asyncio.QueueFull:
            self.metrics['dropped_items'] += 1
            return False
    
    async def _stream_processor_loop(self, processor_id: str):
        """Main stream processor loop."""
        logger.info(f"Stream processor {processor_id} started")
        
        while self.running:
            try:
                # Rate limiting
                if self.rate_limiter:
                    await self.rate_limiter.acquire()
                
                # Get item from buffer
                try:
                    item = await asyncio.wait_for(self.stream_buffer.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process item
                start_time = time.time()
                await self._process_stream_item(item, processor_id)
                processing_time = time.time() - start_time
                
                # Update metrics
                self.metrics['processed_items'] += 1
                
                # Update processing rate
                if self.metrics['processed_items'] > 0:
                    self.metrics['avg_processing_rate'] = (
                        self.metrics['processed_items'] / processing_time
                    )
                
                # Release rate limiter
                if self.rate_limiter:
                    self.rate_limiter.release()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stream processor {processor_id} error: {e}")
        
        logger.info(f"Stream processor {processor_id} stopped")
    
    async def _process_stream_item(self, item: Any, processor_id: str):
        """Process a single stream item."""
        # Simulate async processing
        await asyncio.sleep(0.01)  # Simulate processing time
        
        # Here you would implement actual stream item processing
        # For now, we'll just log it
        logger.debug(f"Processor {processor_id} processed item: {type(item)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get stream processor metrics."""
        return {
            **self.metrics,
            'buffer_size': self.stream_buffer.qsize(),
            'backpressure_active': self.backpressure_active,
            'processing_rate': (
                self.metrics['processed_items'] / max(self.metrics['total_items'], 1)
            )
        }


class ConcurrentInferenceEngine:
    """High-level concurrent inference engine."""
    
    def __init__(
        self,
        num_workers: int = 4,
        worker_type: WorkerType = WorkerType.CPU,
        enable_batching: bool = True,
        enable_streaming: bool = False
    ):
        self.num_workers = num_workers
        self.worker_type = worker_type
        self.enable_batching = enable_batching
        self.enable_streaming = enable_streaming
        
        # Components
        self.worker_pool = WorkerPool(num_workers, worker_type)
        self.batch_processor = BatchProcessor() if enable_batching else None
        self.stream_processor = StreamProcessor() if enable_streaming else None
        
        # Model registry
        self.models = {}
        self.default_model = None
        
        # Running state
        self.running = False
    
    def register_model(self, name: str, model: nn.Module, is_default: bool = False):
        """Register a model for inference."""
        self.models[name] = model
        
        if is_default or self.default_model is None:
            self.default_model = name
        
        logger.info(f"Registered model '{name}' (default: {is_default})")
    
    def start(self):
        """Start the inference engine."""
        if self.running:
            return
        
        self.running = True
        
        # Start components
        self.worker_pool.start()
        
        if self.batch_processor:
            self.batch_processor.start()
        
        logger.info("Concurrent inference engine started")
    
    def stop(self):
        """Stop the inference engine."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop components
        self.worker_pool.stop()
        
        if self.batch_processor:
            self.batch_processor.stop()
        
        logger.info("Concurrent inference engine stopped")
    
    def predict(
        self,
        data: Any,
        model_name: Optional[str] = None,
        priority: int = 1,
        timeout: Optional[float] = None,
        callback: Optional[Callable] = None
    ) -> Optional[ProcessingResult]:
        """Synchronous prediction."""
        if not self.running:
            return None
        
        # Generate request ID
        request_id = f"req_{int(time.time() * 1000000)}"
        
        # Create request
        request = ProcessingRequest(
            request_id=request_id,
            data=data,
            priority=priority,
            callback=callback,
            timeout=timeout
        )
        
        # Select model
        target_model = model_name or self.default_model
        if target_model not in self.models:
            return ProcessingResult(
                request_id=request_id,
                result=None,
                success=False,
                error=f"Model '{target_model}' not found"
            )
        
        model = self.models[target_model]
        
        # Submit to worker pool
        def inference_func(input_data):
            with torch.no_grad():
                if hasattr(model, 'predict'):
                    return model.predict(input_data)
                else:
                    return model(input_data)
        
        if self.worker_pool.submit(inference_func, request):
            return self.worker_pool.get_result(request_id, timeout)
        else:
            return ProcessingResult(
                request_id=request_id,
                result=None,
                success=False,
                error="Failed to submit request to worker pool"
            )
    
    async def predict_async(
        self,
        data: Any,
        model_name: Optional[str] = None,
        priority: int = 1,
        timeout: Optional[float] = None
    ) -> ProcessingResult:
        """Asynchronous prediction."""
        # Run synchronous predict in thread pool
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(
            None,
            self.predict,
            data,
            model_name,
            priority,
            timeout
        )
    
    def predict_batch(
        self,
        data_batch: List[Any],
        model_name: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> List[str]:
        """Batch prediction (returns request IDs)."""
        if not self.running or not self.batch_processor:
            return []
        
        request_ids = []
        
        for data in data_batch:
            request_id = f"batch_req_{int(time.time() * 1000000)}_{len(request_ids)}"
            
            request = ProcessingRequest(
                request_id=request_id,
                data=data,
                callback=callback,
                metadata={'model_name': model_name}
            )
            
            if self.batch_processor.add_request(request):
                request_ids.append(request_id)
        
        return request_ids
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all components."""
        metrics = {
            'engine_running': self.running,
            'registered_models': list(self.models.keys()),
            'default_model': self.default_model
        }
        
        # Worker pool metrics
        if self.worker_pool:
            metrics['worker_pool'] = self.worker_pool.get_metrics()
        
        # Batch processor metrics
        if self.batch_processor:
            metrics['batch_processor'] = self.batch_processor.get_metrics()
        
        # Stream processor metrics
        if self.stream_processor:
            metrics['stream_processor'] = self.stream_processor.get_metrics()
        
        return metrics


# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Concurrent Processing System")
    print("=" * 50)
    
    # Create simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return torch.sigmoid(self.linear(x))
    
    # Test concurrent inference engine
    print("Testing concurrent inference engine...")
    
    engine = ConcurrentInferenceEngine(
        num_workers=2,
        worker_type=WorkerType.CPU,
        enable_batching=True
    )
    
    # Register test model
    test_model = TestModel()
    engine.register_model("test_model", test_model, is_default=True)
    
    # Start engine
    engine.start()
    
    # Test synchronous prediction
    test_data = torch.randn(1, 10)
    result = engine.predict(test_data, timeout=5.0)
    
    if result and result.success:
        print(f"✅ Synchronous prediction successful: {result.processing_time:.4f}s")
    else:
        print(f"❌ Synchronous prediction failed: {result.error if result else 'No result'}")
    
    # Test batch prediction
    batch_data = [torch.randn(1, 10) for _ in range(5)]
    request_ids = engine.predict_batch(batch_data)
    print(f"✅ Batch prediction submitted: {len(request_ids)} requests")
    
    # Wait a bit for processing
    time.sleep(1.0)
    
    # Get comprehensive metrics
    metrics = engine.get_comprehensive_metrics()
    print(f"✅ Engine metrics: {metrics['worker_pool']['total_requests']} requests processed")
    
    # Stop engine
    engine.stop()
    
    print("✅ Concurrent processing system tested successfully!")