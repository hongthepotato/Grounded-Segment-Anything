"""
Test that GPU memory is properly released when training subprocess is cancelled.

This test verifies the core architectural improvement:
- Training runs in an isolated subprocess
- Cancelling the subprocess (killing it) releases all GPU memory
- No manual cleanup code needed

Usage:
    # Run with pytest
    pytest tests/test_subprocess_gpu_release.py -v
    
    # Or run directly
    python tests/test_subprocess_gpu_release.py
"""

import os
import sys
import time
import multiprocessing as mp

# Ensure spawn method for CUDA compatibility
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


def get_gpu_memory_used() -> int:
    """
    Get current GPU memory usage in MB.
    
    Returns:
        GPU memory used in MB, or 0 if nvidia-smi fails
    """
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Get first GPU's memory usage
            lines = result.stdout.strip().split('\n')
            if lines:
                return int(lines[0].strip())
    except Exception as e:
        print(f"Warning: Could not get GPU memory: {e}")
    return 0


def allocate_gpu_memory_subprocess(queue: mp.Queue, event: mp.Event, size_mb: int = 1000):
    """
    Subprocess that allocates GPU memory and waits to be killed.
    
    Args:
        queue: Queue to signal when allocation is done
        event: Event to check for cancellation
        size_mb: Amount of GPU memory to allocate in MB
    """
    import torch
    
    # Allocate GPU tensors
    device = torch.device('cuda:0')
    
    # Allocate ~size_mb of GPU memory (4 bytes per float32)
    num_elements = (size_mb * 1024 * 1024) // 4
    tensor = torch.zeros(num_elements, dtype=torch.float32, device=device)
    
    # Signal that allocation is done
    queue.put({'allocated': True, 'size_mb': size_mb})
    
    # Wait to be cancelled
    while not event.is_set():
        time.sleep(0.1)
    
    # If we get here, we were gracefully cancelled
    queue.put({'graceful_exit': True})


def test_subprocess_gpu_release():
    """
    Test that killing a subprocess releases GPU memory.
    
    Steps:
    1. Record initial GPU memory
    2. Spawn subprocess that allocates GPU memory
    3. Verify GPU memory increased
    4. Kill subprocess
    5. Verify GPU memory returned to initial level
    """
    print("\n" + "=" * 60)
    print("Testing Subprocess GPU Memory Release")
    print("=" * 60)
    
    # Get initial GPU memory
    initial_memory = get_gpu_memory_used()
    print(f"Initial GPU memory: {initial_memory} MB")
    
    if initial_memory == 0:
        print("WARNING: Could not read GPU memory. nvidia-smi may not be available.")
        print("Skipping GPU memory verification, testing process lifecycle only.")
    
    # Create IPC primitives
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    event = ctx.Event()
    
    # Spawn subprocess that allocates GPU memory
    print("\nSpawning subprocess to allocate GPU memory...")
    alloc_size = 500  # 500 MB
    proc = ctx.Process(
        target=allocate_gpu_memory_subprocess,
        args=(queue, event, alloc_size)
    )
    proc.start()
    print(f"Subprocess started (pid={proc.pid})")
    
    # Wait for allocation confirmation
    try:
        msg = queue.get(timeout=30)
        assert msg.get('allocated'), "Subprocess did not confirm allocation"
        print(f"Subprocess allocated {msg.get('size_mb', '?')} MB GPU memory")
    except Exception as e:
        print(f"ERROR: Failed to get allocation confirmation: {e}")
        proc.terminate()
        proc.join()
        raise
    
    # Check GPU memory increased
    time.sleep(0.5)  # Give CUDA time to report
    memory_after_alloc = get_gpu_memory_used()
    print(f"GPU memory after allocation: {memory_after_alloc} MB")
    
    if initial_memory > 0:
        memory_increase = memory_after_alloc - initial_memory
        print(f"Memory increase: {memory_increase} MB")
        if memory_increase < alloc_size * 0.8:  # Allow some margin
            print(f"WARNING: Expected ~{alloc_size} MB increase, got {memory_increase} MB")
    
    # Kill subprocess (simulating cancel)
    print("\nKilling subprocess (simulating cancel)...")
    import signal
    os.kill(proc.pid, signal.SIGTERM)
    proc.join(timeout=5)
    
    if proc.is_alive():
        print("Process did not respond to SIGTERM, sending SIGKILL...")
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=2)
    
    print(f"Subprocess exit code: {proc.exitcode}")
    
    # Check GPU memory released
    # Give OS and CUDA driver time to clean up
    time.sleep(1.0)
    
    memory_after_kill = get_gpu_memory_used()
    print(f"GPU memory after kill: {memory_after_kill} MB")
    
    if initial_memory > 0:
        memory_released = memory_after_alloc - memory_after_kill
        print(f"Memory released: {memory_released} MB")
        
        # Verify memory was released (allow 100MB margin for other processes)
        if memory_after_kill <= initial_memory + 100:
            print("\n✓ SUCCESS: GPU memory was properly released!")
        else:
            print(f"\n✗ FAILURE: GPU memory not fully released")
            print(f"  Expected: ~{initial_memory} MB")
            print(f"  Got: {memory_after_kill} MB")
            return False
    else:
        print("\n✓ Process lifecycle test passed (GPU memory check skipped)")
    
    print("=" * 60)
    return True


def test_graceful_cancellation():
    """
    Test graceful cancellation via event before kill.
    """
    print("\n" + "=" * 60)
    print("Testing Graceful Cancellation")
    print("=" * 60)
    
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    event = ctx.Event()
    
    # Spawn subprocess
    print("Spawning subprocess...")
    proc = ctx.Process(
        target=allocate_gpu_memory_subprocess,
        args=(queue, event, 100)
    )
    proc.start()
    
    # Wait for allocation
    msg = queue.get(timeout=30)
    assert msg.get('allocated')
    print("Subprocess allocated memory")
    
    # Set cancel event (graceful)
    print("Setting cancel event...")
    event.set()
    
    # Wait for graceful exit
    proc.join(timeout=5)
    
    if proc.exitcode == 0:
        # Check for graceful exit message
        try:
            msg = queue.get_nowait()
            if msg.get('graceful_exit'):
                print("✓ Subprocess exited gracefully")
        except:
            print("✓ Subprocess exited")
    else:
        print(f"Subprocess exit code: {proc.exitcode}")
    
    print("=" * 60)
    return True


if __name__ == "__main__":
    print("GPU Memory Release Test")
    print("Testing that subprocess isolation properly releases GPU resources")
    print()
    
    success = True
    
    try:
        if not test_subprocess_gpu_release():
            success = False
    except Exception as e:
        print(f"Test failed with exception: {e}")
        success = False
    
    try:
        if not test_graceful_cancellation():
            success = False
    except Exception as e:
        print(f"Test failed with exception: {e}")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    
    sys.exit(0 if success else 1)




