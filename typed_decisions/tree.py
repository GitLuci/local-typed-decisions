"""Causal attention tree: shared prefix, independent question branches."""
from dataclasses import dataclass


@dataclass
class PackedTree:
    ids: list[int]
    positions: list[int]
    prefix_length: int
    branches: list[tuple[int, int]]

    @property
    def readout_positions(self):
        return [end-1 for _, end in self.branches]

    def mask(self, torch, dtype):
        """Additive 4-D mask. No branch may read another branch, even an earlier one."""
        n = len(self.ids)
        allowed = torch.zeros((n, n), dtype=torch.bool)
        p = self.prefix_length
        allowed[:p, :p] = torch.ones((p, p), dtype=torch.bool).tril()
        for start, end in self.branches:
            allowed[start:end, :p] = True
            allowed[start:end, start:end] = torch.ones((end-start, end-start), dtype=torch.bool).tril()
        mask = torch.full((n, n), float("-inf"), dtype=dtype)
        mask.masked_fill_(allowed, 0)
        return mask[None, None]


def pack_tree(sequences, prefix_length, max_tokens=4096):
    if not sequences or prefix_length < 0:
        raise ValueError("nonempty token sequences and a nonnegative prefix length are required")
    prefix = sequences[0][:prefix_length]
    if any(len(seq) <= prefix_length or seq[:prefix_length] != prefix for seq in sequences):
        raise ValueError("each branch must share exactly the specified prefix and have a suffix")
    ids = list(prefix)
    positions = list(range(prefix_length))
    spans = []
    for seq in sequences:
        start = len(ids)
        ids.extend(seq[prefix_length:])
        positions.extend(range(prefix_length, len(seq)))
        spans.append((start, len(ids)))
    if len(ids) > max_tokens:
        raise ValueError(f"packed request exceeds {max_tokens} tokens; split the question set")
    return PackedTree(ids, positions, prefix_length, spans)


def common_prefix_length(sequences, limit):
    length = 0
    for i in range(min(limit, min(map(len, sequences))-1)):
        if any(seq[i] != sequences[0][i] for seq in sequences[1:]):
            break
        length += 1
    return length
