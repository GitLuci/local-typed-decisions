"""Compute shared-prefix attention without a dense cross-question attention matrix.

Each layer projects and transforms the shared state once. A prefix SDPA and one
batched branch SDPA then reuse those keys/values. This is one backbone traversal,
not a cache prefill followed by one model forward per question.
"""
import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
from transformers.masking_utils import sdpa_mask

BACKEND = "shared_prefix_tree"


def tree_attention_forward(module, query, key, value, attention_mask,
                           tree_layout=None, tree_capture=None, tree_cached_prefix=None, tree_bucket_branches=False,
                           dropout=0., scaling=None, **kwargs):
    if tree_capture is not None:
        if module.training or tree_layout is not None:
            raise ValueError("prefix capture requires an eval-mode ordinary prefix forward")
        tree_capture[module.layer_idx] = (key.detach().clone(), value.detach().clone())
    if tree_layout is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)
    if query.shape[0] != 1 or key.shape[-2] != len(tree_layout.ids):
        raise ValueError("tree attention requires one packed, uncached request")
    if kwargs.get("sliding_window") is not None:
        raise ValueError("sliding-window attention is not implemented for tree layout")
    p = tree_layout.prefix_length
    if tree_cached_prefix is not None and p:
        raise ValueError("cached tree inputs must contain suffix tokens only")
    spans = tree_layout.branches
    n = len(spans)
    lengths = [end-start for start,end in spans]
    longest = max(lengths)
    if n == 1 and tree_cached_prefix is None:
        return sdpa_attention_forward(module, query, key, value, None,
                                      dropout=dropout, scaling=scaling, **kwargs)
    # Same GQA head expansion as the stock CPU SDPA implementation.
    groups = getattr(module, "num_key_value_groups", 1)
    key = repeat_kv(key, groups)
    value = repeat_kv(value, groups)
    outputs = []
    if p:
        outputs.append(F.scaled_dot_product_attention(query[:, :, :p], key[:, :, :p], value[:, :, :p],
                                                      dropout_p=dropout, is_causal=True, scale=scaling))
    # Copies of shared K/V are materialized for the batched CPU kernel; projections,
    # state attention and state MLP are still computed once. This is not zero-copy KV.
    prefix_key, prefix_value = key[:, :, :p], value[:, :, :p]
    if tree_cached_prefix is not None:
        prefix_key, prefix_value = tree_cached_prefix[module.layer_idx]
        prefix_key = repeat_kv(prefix_key, groups)
        prefix_value = repeat_kv(prefix_value, groups)
        p = prefix_key.shape[-2]
    groups = [list(range(n))]
    if tree_bucket_branches and n*longest > 1.5*sum(lengths):
        groups=[]
        for i in sorted(range(n),key=lengths.__getitem__):
            if not groups or lengths[i]>2*lengths[groups[-1][0]]:
                groups.append([])
            groups[-1].append(i)
    branch_outputs={}
    for indices in groups:
        width=max(lengths[i] for i in indices)
        def pack_branches(tensor):
            return torch.cat([F.pad(tensor[:,:,spans[i][0]:spans[i][1]], (0,0,0,width-lengths[i])) for i in indices],dim=0)
        branch_query=pack_branches(query)
        branch_key=torch.cat([prefix_key.expand(len(indices),-1,-1,-1),pack_branches(key)],dim=2)
        branch_value=torch.cat([prefix_value.expand(len(indices),-1,-1,-1),pack_branches(value)],dim=2)
        q_index=torch.arange(width,device=query.device)[None,:,None]
        k_index=torch.arange(p+width,device=query.device)[None,None,:]
        valid_length=torch.tensor([lengths[i] for i in indices],device=query.device)[:,None,None]
        allowed=(k_index<=p+q_index)&(k_index<p+valid_length)
        branches=F.scaled_dot_product_attention(branch_query,branch_key,branch_value,
                                                attn_mask=allowed[:,None],dropout_p=dropout,scale=scaling)
        for row,i in enumerate(indices):
            branch_outputs[i]=branches[row:row+1,:,:lengths[i]]
    outputs.extend(branch_outputs[i] for i in range(n))
    output = torch.cat(outputs, dim=2).transpose(1,2).contiguous()
    return output, None


def register_tree_attention():
    AttentionInterface.register(BACKEND, tree_attention_forward)
    # Keep the stock mask builder for ordinary reference calls without tree_layout.
    # Tree calls supply a mapping to bypass full dense-mask allocation; the kernel
    # above enforces all causal, branch-isolation and padding rules explicitly.
    AttentionMaskInterface.register(BACKEND, sdpa_mask)
