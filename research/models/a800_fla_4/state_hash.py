"""CPU-only full-tensor byte fingerprints for exact-state replay checks."""
import ctypes,ctypes.util,hashlib
from pathlib import Path

class Hash128(ctypes.Structure):
    _fields_=[('low64',ctypes.c_uint64),('high64',ctypes.c_uint64)]

_name=ctypes.util.find_library('xxhash')
if _name:
    _lib=ctypes.CDLL(_name)
    _hash=_lib.XXH3_128bits
    _hash.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    _hash.restype=Hash128
    _lib.XXH_versionNumber.restype=ctypes.c_uint
    paths={l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines() if 'libxxhash' in l}
    ENGINE=dict(name='XXH3_128',version=int(_lib.XXH_versionNumber()),libraries={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
else:
    ENGINE=dict(name='SHA256')

def fingerprint(x):
    if x is None:return None
    data=x.detach().cpu().contiguous().numpy()
    if _name:
        h=_hash(data.ctypes.data,data.nbytes)
        return f'{h.high64:016x}{h.low64:016x}'
    return hashlib.sha256(memoryview(data).cast('B')).hexdigest()
