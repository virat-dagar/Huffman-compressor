import heapq
import json
import os
import sys
import time
from collections import Counter


class Node:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freq):
    """Build the Huffman tree from a character frequency table. O(k log k) for k distinct chars."""
    heap = [Node(ch, f) for ch, f in freq.items()]
    heapq.heapify(heap)

    # Edge case: only one distinct character
    if len(heap) == 1:
        only = heap[0]
        return Node(None, only.freq, left=only)

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, Node(None, a.freq + b.freq, a, b))
    return heap[0]


def generate_codes(node, prefix="", codes=None):
    """Walk the tree: left = '0', right = '1'. Returns {char: binary_string}."""
    if codes is None:
        codes = {}
    if node.char is not None:
        codes[node.char] = prefix or "0"
    else:
        if node.left:
            generate_codes(node.left, prefix + "0", codes)
        if node.right:
            generate_codes(node.right, prefix + "1", codes)
    return codes


def compress(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text:
        raise ValueError("Input file is empty.")

    freq = Counter(text)  # O(n)
    root = build_huffman_tree(freq)  # O(k log k)
    codes = generate_codes(root)  # O(k)

    bits = "".join(codes[ch] for ch in text)  # O(n)
    pad = (8 - len(bits) % 8) % 8  # pad to a whole number of bytes
    bits += "0" * pad
    data = bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))

    # Header stores the code table so the decoder can rebuild the mapping
    header = json.dumps({"codes": codes, "pad": pad}).encode("utf-8")
    with open(out_path, "wb") as f:
        f.write(len(header).to_bytes(4, "big"))
        f.write(header)
        f.write(data)


def decompress(in_path, out_path):
    with open(in_path, "rb") as f:
        header_len = int.from_bytes(f.read(4), "big")
        header = json.loads(f.read(header_len).decode("utf-8"))
        data = f.read()

    reverse = {code: ch for ch, code in header["codes"].items()}
    bits = "".join(f"{byte:08b}" for byte in data)
    if header["pad"]:
        bits = bits[: -header["pad"]]

    out, current = [], ""
    for bit in bits:  # O(n) prefix-free decoding
        current += bit
        if current in reverse:
            out.append(reverse[current])
            current = ""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(out))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python huffman.py <input_text_file>")
        sys.exit(1)

    src = sys.argv[1]
    compressed = src + ".huff"
    restored = src + ".restored.txt"

    start = time.perf_counter()
    compress(src, compressed)
    comp_time = time.perf_counter() - start

    start = time.perf_counter()
    decompress(compressed, restored)
    decomp_time = time.perf_counter() - start

    with open(src, "r", encoding="utf-8") as a, open(
        restored, "r", encoding="utf-8"
    ) as b:
        lossless = a.read() == b.read()

    orig_size = os.path.getsize(src)
    comp_size = os.path.getsize(compressed)

    print(f"Original size:    {orig_size} bytes")
    print(f"Compressed size:  {comp_size} bytes")
    print(f"Compression ratio: {comp_size / orig_size:.2%}")
    print(f"Compression time:   {comp_time:.6f} s")
    print(f"Decompression time: {decomp_time:.6f} s")
    print(f"Lossless round trip: {lossless}")
