"""dense8s CUBIN update, offline: no training, calibration or TensorRT rebuild."""
import base64
import hashlib
import importlib.util
import json
import lzma
import zlib
from pathlib import Path

PAYLOADS = {}
HASHES = {'repack_dense8s_runtime.py': 'd20e04dcb7960f3aaebdccac06edb7e4e99ed1b38ef3ff7e66a3fe681ea78377', 'cudart.py': '486f4da72b8fc7f2ce76a8bcdcd1da9d4361606311eed4c3a9c2212800e75633', 'dense8s_kernels.cu': '06e6d84c6e106f9a9fc2646591137c685ff5e078b66fd27bef4caecc9a784f37', 'kernel-manifest.json': 'e1690d433376139b398f1ef99a29c838731987834389ea4f1616980cf1cc7655', 'dense8s_sm75.cubin': 'ca90f89df47eea08cc35bff84d39b693f2da3a57fe3bd851a80bf5c9707a296c'}
root = Path("/kaggle/working/repack-runtime-src")
root.mkdir(parents=True, exist_ok=True)
for name, encoded in PAYLOADS.items():
    raw = (lzma.decompress(base64.b64decode(encoded)) if name == 'dense8s_sm75.cubin'
           else zlib.decompress(base64.b64decode(encoded)))
    if hashlib.sha256(raw).hexdigest() != HASHES[name]:
        raise RuntimeError(f"hash of the embedded source differs: {name}")
    (root / name).write_bytes(raw)
spec = importlib.util.spec_from_file_location("dense8s_repack", root / "repack_dense8s_runtime.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
packs = helper.discover_packs(Path("/kaggle/input"), prefix='d8s376x2j')
runtime = root / "kernel-manifest.json" if True else None
result = helper.run(packs, Path("/kaggle/working/dense8s-runtime"),
                    source=root / "dense8s_kernels.cu", cudart_path=root / "cudart.py",
                    runtime=runtime, verify_load=False)
print(json.dumps(result, indent=2), flush=True)

TILE_SOURCES = {}
for name, encoded in TILE_SOURCES.items():
    (Path("/kaggle/working/dense8s-runtime") / name).write_bytes(zlib.decompress(base64.b64decode(encoded)))
