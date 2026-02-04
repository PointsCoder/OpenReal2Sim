"""
Detailed timing instrumentation for the randomized rollout pipeline.
Enable/disable with ENABLE_DETAILED_TIMING flag.

Usage:
    from timing_utils import Timer, ENABLE_DETAILED_TIMING, print_timing_summary
    
    # Option 1: Context manager
    with Timer("operation_name"):
        do_something()
    
    # Option 2: Decorator
    @timed("function_name")
    def my_function():
        pass
    
    # Option 3: Manual start/stop
    timer = Timer("operation")
    timer.start()
    do_something()
    timer.stop()
    
    # Print summary at the end
    print_timing_summary()
"""

import time
import functools
from collections import defaultdict
from typing import Optional, Dict, List
import threading

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL TIMING FLAG - Set to False to disable all timing overhead
# ═══════════════════════════════════════════════════════════════════════════════
ENABLE_DETAILED_TIMING = False  # Disabled to prevent memory accumulation and overhead

# Thread-safe storage for timing data
_timing_lock = threading.Lock()
_timing_data: Dict[str, List[float]] = defaultdict(list)
_timing_hierarchy: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
_current_parent: Optional[str] = None
_call_counts: Dict[str, int] = defaultdict(int)


class Timer:
    """
    High-resolution timer with hierarchical tracking support.
    
    Features:
    - Automatic parent-child relationship tracking
    - Cumulative statistics (min, max, mean, total, count)
    - Thread-safe operation
    - Zero overhead when ENABLE_DETAILED_TIMING is False
    """
    
    def __init__(self, name: str, parent: Optional[str] = None, log_immediately: bool = True):
        """
        Args:
            name: Timer name (should be descriptive, e.g., "motion_planning_batch")
            parent: Optional parent timer name for hierarchical tracking
            log_immediately: If True, prints timing info when timer stops
        """
        self.name = name
        self.parent = parent
        self.log_immediately = log_immediately
        self._start_time: Optional[float] = None
        self._elapsed: float = 0.0
        self._previous_parent: Optional[str] = None
    
    def start(self) -> 'Timer':
        """Start the timer. Returns self for chaining."""
        if not ENABLE_DETAILED_TIMING:
            return self
        self._start_time = time.perf_counter()
        global _current_parent
        self._previous_parent = _current_parent
        _current_parent = self.name
        return self
    
    def stop(self) -> float:
        """Stop the timer and record the elapsed time. Returns elapsed seconds."""
        if not ENABLE_DETAILED_TIMING:
            return 0.0
        
        if self._start_time is None:
            return 0.0
        
        self._elapsed = time.perf_counter() - self._start_time
        
        with _timing_lock:
            _timing_data[self.name].append(self._elapsed)
            _call_counts[self.name] += 1
            
            # Track hierarchy
            if self.parent or self._previous_parent:
                parent_name = self.parent or self._previous_parent
                _timing_hierarchy[parent_name][self.name].append(self._elapsed)
        
        global _current_parent
        _current_parent = self._previous_parent
        
        if self.log_immediately:
            print(f"[TIMING] {self.name}: {self._elapsed*1000:.2f}ms")
        
        self._start_time = None
        return self._elapsed
    
    def __enter__(self) -> 'Timer':
        """Context manager entry."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return self._elapsed
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self._elapsed * 1000


def timed(name: str, log_immediately: bool = True):
    """
    Decorator to time a function.
    
    Usage:
        @timed("my_function")
        def my_function():
            pass
    """
    def decorator(func):
        if not ENABLE_DETAILED_TIMING:
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(name, log_immediately=log_immediately):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class TimingBlock:
    """
    Context manager for timing a block of code with sub-timers.
    
    Usage:
        with TimingBlock("main_operation") as tb:
            with tb.sub("sub_operation_1"):
                do_something()
            with tb.sub("sub_operation_2"):
                do_something_else()
    """
    
    def __init__(self, name: str, log_immediately: bool = True):
        self.name = name
        self.log_immediately = log_immediately
        self._timer = Timer(name, log_immediately=False)
        self._sub_timers: List[Timer] = []
    
    def __enter__(self) -> 'TimingBlock':
        self._timer.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = self._timer.stop()
        if self.log_immediately and ENABLE_DETAILED_TIMING:
            print(f"[TIMING] ══════════════════════════════════════")
            print(f"[TIMING] {self.name} TOTAL: {elapsed*1000:.2f}ms")
            for st in self._sub_timers:
                pct = (st.elapsed / elapsed * 100) if elapsed > 0 else 0
                print(f"[TIMING]   ├─ {st.name}: {st.elapsed*1000:.2f}ms ({pct:.1f}%)")
            print(f"[TIMING] ══════════════════════════════════════")
    
    def sub(self, name: str) -> Timer:
        """Create a sub-timer within this block."""
        timer = Timer(f"{self.name}/{name}", parent=self.name, log_immediately=False)
        self._sub_timers.append(timer)
        return timer


def log_timing(name: str, elapsed_seconds: float) -> None:
    """Manually log a timing measurement."""
    if not ENABLE_DETAILED_TIMING:
        return
    
    with _timing_lock:
        _timing_data[name].append(elapsed_seconds)
        _call_counts[name] += 1
    
    print(f"[TIMING] {name}: {elapsed_seconds*1000:.2f}ms")


def get_timing_stats(name: str) -> Dict[str, float]:
    """Get timing statistics for a named timer."""
    with _timing_lock:
        times = _timing_data.get(name, [])
    
    if not times:
        return {"count": 0, "total": 0, "mean": 0, "min": 0, "max": 0}
    
    return {
        "count": len(times),
        "total": sum(times),
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
    }


def print_timing_summary() -> None:
    """Print a comprehensive summary of all recorded timings."""
    if not ENABLE_DETAILED_TIMING:
        print("[TIMING] Timing is disabled (ENABLE_DETAILED_TIMING=False)")
        return
    
    with _timing_lock:
        data = dict(_timing_data)
        counts = dict(_call_counts)
    
    if not data:
        print("[TIMING] No timing data recorded.")
        return
    
    print("\n")
    print("╔" + "═" * 98 + "╗")
    print("║" + " DETAILED TIMING SUMMARY ".center(98) + "║")
    print("╠" + "═" * 98 + "╣")
    print("║ {:<40} {:>10} {:>12} {:>12} {:>10} {:>10} ║".format(
        "Operation", "Count", "Total (s)", "Mean (ms)", "Min (ms)", "Max (ms)"))
    print("╠" + "═" * 98 + "╣")
    
    # Sort by total time (descending)
    sorted_data = sorted(data.items(), key=lambda x: sum(x[1]), reverse=True)
    
    for name, times in sorted_data:
        if not times:
            continue
        total = sum(times)
        mean = total / len(times) * 1000
        min_t = min(times) * 1000
        max_t = max(times) * 1000
        count = counts.get(name, len(times))
        
        # Truncate long names
        display_name = name if len(name) <= 40 else "..." + name[-37:]
        
        print("║ {:<40} {:>10} {:>12.3f} {:>12.2f} {:>10.2f} {:>10.2f} ║".format(
            display_name, count, total, mean, min_t, max_t))
    
    print("╚" + "═" * 98 + "╝")
    print("\n")


def reset_timing_data() -> None:
    """Clear all recorded timing data."""
    global _timing_data, _timing_hierarchy, _call_counts, _current_parent
    with _timing_lock:
        _timing_data.clear()
        _timing_hierarchy.clear()
        _call_counts.clear()
        _current_parent = None


def get_detailed_breakdown() -> Dict[str, Dict]:
    """Get detailed breakdown of all timing data including hierarchy."""
    with _timing_lock:
        result = {}
        for name, times in _timing_data.items():
            if not times:
                continue
            result[name] = {
                "count": len(times),
                "total_s": sum(times),
                "mean_ms": sum(times) / len(times) * 1000,
                "min_ms": min(times) * 1000,
                "max_ms": max(times) * 1000,
                "all_times_ms": [t * 1000 for t in times],
            }
        return result
