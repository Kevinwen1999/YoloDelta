import traceback
import time

try:
    import testKalman2 as tk
except Exception as e:
    print('Failed to import testKalman2:', e)
    raise

try:
    print('Building inference backend...')
    backend = tk.build_inference_backend()
    print('Backend created:', getattr(backend, 'name', '<unknown>'))
    print('Warming up backend to trigger any exports/engine builds...')
    backend.warmup()
    print('Warmup complete')
except Exception:
    traceback.print_exc()

print('Done')
