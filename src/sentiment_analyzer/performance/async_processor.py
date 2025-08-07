"""
Asynchronous processing for high-throughput sentiment analysis.
"""
import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
from queue import Queue, Empty
import threading

from ..core.analyzer import SentimentAnalyzer
from ..core.models import AnalysisConfig, ModelType, SentimentResult
from ..security.validation import input_validator, ValidationResult
from ..monitoring.metrics import metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Task for asynchronous processing."""
    id: str
    text: str
    model_type: Optional[ModelType]
    metadata: Dict[str, Any]
    created_at: float
    priority: int = 0  # Lower number = higher priority


@dataclass
class ProcessingResult:
    """Result of asynchronous processing."""
    task_id: str
    result: Optional[SentimentResult]
    error: Optional[str]
    processing_time: float
    validation_result: Optional[ValidationResult]


class WorkerPool:
    """
    Pool of worker threads/processes for sentiment analysis.
    """
    
    def __init__(self, 
                 num_workers: int = None,
                 use_processes: bool = False,
                 analyzer_config: Optional[AnalysisConfig] = None):
        self.num_workers = num_workers or min(8, (mp.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self.analyzer_config = analyzer_config or AnalysisConfig()
        
        # Initialize executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        # Performance tracking
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self._stats_lock = threading.Lock()
        
        logger.info(f"Initialized worker pool: {self.num_workers} {'processes' if use_processes else 'threads'}")
    
    async def process_task(self, task: ProcessingTask) -> ProcessingResult:
        """Process a single task asynchronously."""
        start_time = time.time()
        
        with self._stats_lock:
            self.active_tasks += 1
        
        try:
            loop = asyncio.get_event_loop()
            
            # Run in executor to avoid blocking
            result = await loop.run_in_executor(
                self.executor,
                self._worker_function,
                task
            )
            
            with self._stats_lock:
                self.completed_tasks += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            
            with self._stats_lock:
                self.failed_tasks += 1
            
            return ProcessingResult(
                task_id=task.id,
                result=None,
                error=str(e),
                processing_time=time.time() - start_time,
                validation_result=None
            )
        finally:
            with self._stats_lock:
                self.active_tasks -= 1
    
    def _worker_function(self, task: ProcessingTask) -> ProcessingResult:
        """Worker function that runs in executor."""
        start_time = time.time()
        
        try:
            # Validate input
            validation_result = input_validator.validate_text(task.text)
            
            if not validation_result.is_valid:
                return ProcessingResult(
                    task_id=task.id,
                    result=None,
                    error=f"Validation failed: {'; '.join(validation_result.warnings)}",
                    processing_time=time.time() - start_time,
                    validation_result=validation_result
                )
            
            # Create analyzer (each worker gets its own instance)
            analyzer = SentimentAnalyzer(self.analyzer_config)
            
            # Perform analysis
            result = analyzer.analyze_text(
                validation_result.sanitized_text,
                model_type=task.model_type
            )
            
            # Record metrics
            metrics_collector.record_analysis(
                model_type=result.model_type.value,
                processing_time=result.processing_time,
                confidence=result.confidence,
                text_length=len(task.text),
                success=True,
                labels=task.metadata
            )
            
            return ProcessingResult(
                task_id=task.id,
                result=result,
                error=None,
                processing_time=time.time() - start_time,
                validation_result=validation_result
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Record error metrics
            model_type = task.model_type or self.analyzer_config.model_type
            metrics_collector.record_analysis(
                model_type=model_type.value,
                processing_time=processing_time,
                confidence=0.0,
                text_length=len(task.text),
                success=False,
                labels=task.metadata
            )
            
            return ProcessingResult(
                task_id=task.id,
                result=None,
                error=str(e),
                processing_time=processing_time,
                validation_result=None
            )
    
    async def process_batch(self, tasks: List[ProcessingTask]) -> List[ProcessingResult]:
        """Process multiple tasks concurrently."""
        if not tasks:
            return []
        
        start_time = time.time()
        
        # Create coroutines for all tasks
        coroutines = [self.process_task(task) for task in tasks]
        
        # Process all tasks concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch task {tasks[i].id} failed: {result}")
                processed_results.append(ProcessingResult(
                    task_id=tasks[i].id,
                    result=None,
                    error=str(result),
                    processing_time=0.0,
                    validation_result=None
                ))
            else:
                processed_results.append(result)
        
        # Record batch metrics
        batch_time = time.time() - start_time
        success_count = sum(1 for r in processed_results if r.error is None)
        error_count = len(processed_results) - success_count
        
        model_type = tasks[0].model_type or self.analyzer_config.model_type
        metrics_collector.record_batch_analysis(
            model_type=model_type.value,
            batch_size=len(tasks),
            total_processing_time=batch_time,
            success_count=success_count,
            error_count=error_count
        )
        
        logger.info(f"Processed batch of {len(tasks)} tasks in {batch_time:.3f}s "
                   f"({success_count} success, {error_count} errors)")
        
        return processed_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        with self._stats_lock:
            return {
                'num_workers': self.num_workers,
                'use_processes': self.use_processes,
                'active_tasks': self.active_tasks,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'total_tasks': self.completed_tasks + self.failed_tasks,
                'success_rate': self.completed_tasks / max(self.completed_tasks + self.failed_tasks, 1)
            }
    
    def shutdown(self, wait: bool = True):
        """Shutdown worker pool."""
        logger.info("Shutting down worker pool...")
        self.executor.shutdown(wait=wait)


class AsyncProcessor:
    """
    High-level asynchronous processor for sentiment analysis.
    """
    
    def __init__(self, 
                 num_workers: int = None,
                 use_processes: bool = False,
                 analyzer_config: Optional[AnalysisConfig] = None,
                 enable_batching: bool = True,
                 batch_size: int = 32,
                 batch_timeout: float = 0.1):
        
        self.worker_pool = WorkerPool(num_workers, use_processes, analyzer_config)
        self.enable_batching = enable_batching
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Task queues
        self.pending_tasks: Queue = Queue()
        self.results_cache: Dict[str, ProcessingResult] = {}
        self.cache_lock = threading.Lock()
        
        # Batching state
        self.batch_processor_running = False
        self.batch_processor_task = None
        
        if enable_batching:
            self._start_batch_processor()
        
        logger.info(f"Initialized async processor with batching={'enabled' if enable_batching else 'disabled'}")
    
    def _start_batch_processor(self):
        """Start background batch processor."""
        if self.batch_processor_running:
            return
        
        self.batch_processor_running = True
        
        async def batch_processor():
            """Background batch processor coroutine."""
            while self.batch_processor_running:
                try:
                    # Collect tasks for batching
                    batch_tasks = []
                    deadline = time.time() + self.batch_timeout
                    
                    # Collect up to batch_size tasks or until timeout
                    while (len(batch_tasks) < self.batch_size and 
                           time.time() < deadline and
                           self.batch_processor_running):
                        
                        try:
                            task = self.pending_tasks.get(timeout=0.01)
                            batch_tasks.append(task)
                        except Empty:
                            continue
                    
                    # Process batch if we have tasks
                    if batch_tasks:
                        results = await self.worker_pool.process_batch(batch_tasks)
                        
                        # Store results
                        with self.cache_lock:
                            for result in results:
                                self.results_cache[result.task_id] = result
                    
                    # Small delay to prevent busy waiting
                    await asyncio.sleep(0.001)
                    
                except Exception as e:
                    logger.error(f"Batch processor error: {e}")
                    await asyncio.sleep(0.1)
        
        # Start batch processor
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.batch_processor_task = loop.create_task(batch_processor())
    
    async def analyze_text_async(self, 
                               text: str, 
                               model_type: Optional[ModelType] = None,
                               task_id: Optional[str] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Analyze text asynchronously."""
        task_id = task_id or f"task_{int(time.time() * 1000000)}"
        metadata = metadata or {}
        
        task = ProcessingTask(
            id=task_id,
            text=text,
            model_type=model_type,
            metadata=metadata,
            created_at=time.time()
        )
        
        if self.enable_batching:
            # Add to batch queue
            self.pending_tasks.put(task)
            
            # Wait for result
            max_wait = 10.0  # Maximum wait time
            start_wait = time.time()
            
            while time.time() - start_wait < max_wait:
                with self.cache_lock:
                    if task_id in self.results_cache:
                        result = self.results_cache.pop(task_id)
                        return result
                
                await asyncio.sleep(0.01)
            
            # Timeout - process directly
            logger.warning(f"Task {task_id} timed out in batch queue, processing directly")
            return await self.worker_pool.process_task(task)
        
        else:
            # Process directly
            return await self.worker_pool.process_task(task)
    
    async def analyze_batch_async(self,
                                texts: List[str],
                                model_type: Optional[ModelType] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> List[ProcessingResult]:
        """Analyze multiple texts asynchronously."""
        metadata = metadata or {}
        
        # Create tasks
        tasks = []
        for i, text in enumerate(texts):
            task = ProcessingTask(
                id=f"batch_{int(time.time() * 1000000)}_{i}",
                text=text,
                model_type=model_type,
                metadata=metadata,
                created_at=time.time()
            )
            tasks.append(task)
        
        # Process batch directly (bypass batching queue for explicit batches)
        return await self.worker_pool.process_batch(tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        worker_stats = self.worker_pool.get_stats()
        
        with self.cache_lock:
            cache_size = len(self.results_cache)
        
        return {
            'worker_pool': worker_stats,
            'batching_enabled': self.enable_batching,
            'batch_size': self.batch_size,
            'batch_timeout': self.batch_timeout,
            'pending_tasks': self.pending_tasks.qsize(),
            'cached_results': cache_size,
            'batch_processor_running': self.batch_processor_running
        }
    
    def shutdown(self):
        """Shutdown async processor."""
        logger.info("Shutting down async processor...")
        
        self.batch_processor_running = False
        
        if self.batch_processor_task:
            try:
                self.batch_processor_task.cancel()
            except Exception as e:
                logger.warning(f"Error canceling batch processor: {e}")
        
        self.worker_pool.shutdown()
        
        with self.cache_lock:
            self.results_cache.clear()


# Global async processor
async_processor = AsyncProcessor(
    num_workers=8,
    use_processes=False,
    enable_batching=True,
    batch_size=32,
    batch_timeout=0.1
)