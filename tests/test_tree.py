import os
os.environ["USE_TF"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pytest
torch = pytest.importorskip("torch")
from transformers import Qwen3Config, Qwen3Model

from typed_decisions.tree import pack_tree

torch.set_num_threads(1)


@pytest.fixture
def backbone():
    torch.manual_seed(123)
    config = Qwen3Config(vocab_size=100, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, attention_dropout=0)
    config._attn_implementation = "sdpa"
    return Qwen3Model(config).eval()


def tree_hidden(model, seqs):
    p = pack_tree(seqs, 3)
    with torch.inference_mode():
        return model(input_ids=torch.tensor([p.ids]), position_ids=torch.tensor([p.positions]),
                     attention_mask=p.mask(torch, torch.float32), use_cache=False).last_hidden_state[0, p.readout_positions]


def test_actual_transformer_matches_independent_causal_paths(backbone):
    sequences = [[1,2,3,10,11], [1,2,3,20,21,22], [1,2,3,30]]
    packed = tree_hidden(backbone, sequences)
    with torch.inference_mode():
        separate = torch.stack([backbone(input_ids=torch.tensor([s]), use_cache=False).last_hidden_state[0,-1] for s in sequences])
    torch.testing.assert_close(packed, separate, atol=1e-6, rtol=1e-5)


def test_earlier_branch_cannot_influence_later_branch(backbone):
    a = tree_hidden(backbone, [[1,2,3,10,11], [1,2,3,20,21,22]])
    b = tree_hidden(backbone, [[1,2,3,98,99,98,99,98], [1,2,3,20,21,22]])
    torch.testing.assert_close(a[1], b[1], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(a[0], b[0])  # positive control: changed branch does change


def test_question_permutation_preserves_representations(backbone):
    seqs = [[1,2,3,10,11], [1,2,3,20,21,22]]
    a = tree_hidden(backbone, seqs)
    b = tree_hidden(backbone, list(reversed(seqs)))
    torch.testing.assert_close(a, b.flip(0), atol=1e-6, rtol=1e-5)


def test_mismatched_prefix_and_capacity_fail():
    with pytest.raises(ValueError):
        pack_tree([[1,2,3], [1,9,4]], 2)
    with pytest.raises(ValueError):
        pack_tree([[1,2,3], [1,2,4]], 2, max_tokens=3)


@pytest.mark.parametrize("prefix", [0, 3])
def test_branched_kernel_matches_dense_outputs_and_gradients(backbone, prefix):
    from typed_decisions.tree_attention import register_tree_attention, BACKEND
    seqs = [[1,2,3,10,11], [1,2,3,20,21,22], [1,2,3,30]]
    packed = pack_tree(seqs, prefix)
    args = {"input_ids": torch.tensor([packed.ids]), "position_ids": torch.tensor([packed.positions]), "use_cache": False}
    dense = backbone(**args, attention_mask=packed.mask(torch, torch.float32)).last_hidden_state
    loss = dense[:, packed.readout_positions].square().sum()
    loss.backward()
    grad = backbone.embed_tokens.weight.grad.clone()
    backbone.zero_grad(set_to_none=True)
    register_tree_attention()
    backbone.set_attn_implementation(BACKEND)
    branched = backbone(**args, attention_mask={"full_attention": None}, tree_layout=packed).last_hidden_state
    branched[:, packed.readout_positions].square().sum().backward()
    torch.testing.assert_close(branched, dense, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(backbone.embed_tokens.weight.grad, grad, atol=1e-4, rtol=1e-3)


def test_reusable_prefix_matches_cold_tree_and_never_changes(backbone):
    from typed_decisions.tree_attention import register_tree_attention, BACKEND
    register_tree_attention()
    backbone.set_attn_implementation(BACKEND)
    sequences = [[1,2,3,10,11], [1,2,3,20,21,22]]
    prefix = [1,2,3]
    cache = {}
    with torch.inference_mode():
        backbone(input_ids=torch.tensor([prefix]), use_cache=False, tree_capture=cache)
        before = {i:tuple(t.clone() for t in pair) for i,pair in cache.items()}
        for seqs in (sequences, list(reversed(sequences)), [[1,2,3,99,98,97], sequences[1]]):
            packed = pack_tree(seqs, 3)
            cold = backbone(input_ids=torch.tensor([packed.ids]), position_ids=torch.tensor([packed.positions]),
                            attention_mask={"full_attention":None}, tree_layout=packed, use_cache=False).last_hidden_state[0,packed.readout_positions]
            suffix = pack_tree([s[3:] for s in seqs], 0)
            warm = backbone(input_ids=torch.tensor([suffix.ids]), position_ids=torch.tensor([[p+3 for p in suffix.positions]]),
                            attention_mask={"full_attention":None}, tree_layout=suffix,
                            tree_cached_prefix=cache, use_cache=False).last_hidden_state[0,suffix.readout_positions]
            torch.testing.assert_close(warm, cold, atol=1e-6, rtol=1e-5)
        for i,pair in cache.items():
            for a,b in zip(pair,before[i]):
                torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_one_branch_fast_path_matches_causal_reference(backbone):
    from typed_decisions.tree_attention import register_tree_attention, BACKEND
    seq = [1,2,3,10,11]
    with torch.inference_mode():
        expected = backbone(input_ids=torch.tensor([seq]), use_cache=False).last_hidden_state
        register_tree_attention()
        backbone.set_attn_implementation(BACKEND)
        for p in (0,3):
            packed = pack_tree([seq], p)
            actual = backbone(input_ids=torch.tensor([packed.ids]), position_ids=torch.tensor([packed.positions]),
                              attention_mask={"full_attention":None}, tree_layout=packed, use_cache=False).last_hidden_state
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


def test_length_buckets_preserve_outputs_and_gradients(backbone):
    from typed_decisions.tree_attention import register_tree_attention,BACKEND
    seqs=[[1,2,3,10], [1,2,3]+[20,21]*12, [1,2,3,30,31], [1,2,3,40]]
    p=pack_tree(seqs,3)
    register_tree_attention(); backbone.set_attn_implementation(BACKEND)
    args={'input_ids':torch.tensor([p.ids]),'position_ids':torch.tensor([p.positions]),
          'attention_mask':{'full_attention':None},'tree_layout':p,'use_cache':False}
    outputs=[]; grads=[]
    for bucket in (False,True):
        backbone.zero_grad(set_to_none=True)
        out=backbone(**args,tree_bucket_branches=bucket).last_hidden_state
        out[:,p.readout_positions].square().sum().backward()
        outputs.append(out.detach()); grads.append(backbone.embed_tokens.weight.grad.clone())
    torch.testing.assert_close(outputs[0],outputs[1],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(grads[0],grads[1],atol=1e-4,rtol=1e-3)
