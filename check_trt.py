import importlib.util, os
print('tensorrt_libs spec:', importlib.util.find_spec('tensorrt_libs'))
try:
    import onnxruntime as ort
    print('onnxruntime providers:', ort.get_available_providers())
except Exception as e:
    print('onnxruntime import error:', e)
print('PATH sample:', os.environ.get('PATH','')[:400])
