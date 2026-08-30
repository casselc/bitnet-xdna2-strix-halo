#!/usr/bin/env python3
"""Extract one I2_S tensor from a GGUF into raw files for the C tests.

The point is to validate our unpacker against REAL BitNet weights, not just
against our own reimplementation of the packer. A round-trip test can agree
with itself while both halves are wrong; real data cannot.

Emits <out>.packed (the raw I2_S blob, including the trailing f32 scale) and
prints the metadata the C side needs.
"""
import struct, sys, json

GT_SIZE = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
GT_FMT  = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}

class Reader:
    def __init__(self, f): self.f = f
    def raw(self, n): return self.f.read(n)
    def u32(self): return struct.unpack('<I', self.f.read(4))[0]
    def u64(self): return struct.unpack('<Q', self.f.read(8))[0]
    def s(self):
        n = self.u64(); return self.f.read(n).decode('utf-8', 'replace')
    def val(self, t):
        if t == 8: return self.s()
        if t == 9:
            et, n = self.u32(), self.u64()
            if et == 8:
                return [self.s() for _ in range(n)]
            self.f.read(GT_SIZE[et] * n); return f'<{n} x type{et}>'
        return struct.unpack(GT_FMT[t], self.f.read(GT_SIZE[t]))[0]

def main():
    path, want, out = sys.argv[1], sys.argv[2], sys.argv[3]
    f = open(path, 'rb')
    r = Reader(f)
    assert r.raw(4) == b'GGUF', 'not a GGUF file'
    ver, n_tensors, n_kv = r.u32(), r.u64(), r.u64()

    kv = {}
    for _ in range(n_kv):
        k = r.s(); t = r.u32(); kv[k] = r.val(t)
    align = kv.get('general.alignment', 32)

    tensors = []
    for _ in range(n_tensors):
        name = r.s(); nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ttype = r.u32(); offset = r.u64()
        tensors.append((name, dims, ttype, offset))

    data_start = f.tell()
    if data_start % align: data_start += align - (data_start % align)

    hit = next((t for t in tensors if t[0] == want), None)
    if hit is None:
        print(f'tensor {want!r} not found. Available (first 12):', file=sys.stderr)
        for t in tensors[:12]: print('  ', t[0], t[1], 'type', t[2], file=sys.stderr)
        sys.exit(1)

    name, dims, ttype, offset = hit
    assert ttype == 36, f'expected GGML_TYPE_I2_S (36), got {ttype}'
    K, N = int(dims[0]), int(dims[1])

    # I2_S blob = K*N/4 packed bytes, then one f32 scale, then 32B alignment pad.
    packed_bytes = (K * N) // 4
    blob_bytes = packed_bytes + 4

    f.seek(data_start + offset)
    blob = f.read(blob_bytes)
    assert len(blob) == blob_bytes, 'short read'
    scale = struct.unpack('<f', blob[packed_bytes:packed_bytes+4])[0]

    with open(out + '.packed', 'wb') as g: g.write(blob)
    meta = {'tensor': name, 'K': K, 'N': N, 'ggml_type': ttype,
            'packed_bytes': packed_bytes, 'scale': scale,
            'file_offset': data_start + offset}
    with open(out + '.json', 'w') as g: json.dump(meta, g, indent=2)
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
