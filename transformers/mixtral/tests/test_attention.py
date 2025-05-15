import torch
from attention import AttentionWithKVCache

def test_attention_output_shape():
    model = AttentionWithKVCache(dim=64, num_heads=8, window_size=5)
    x = torch.randn(2, 10, 64)  # (batch, seq_len, dim)
    out = model(x)
    assert out.shape == (2, 10, 64), "Output shape mismatch"
    

def test_training_windowed_attention():
    model = AttentionWithKVCache(dim=64, num_heads=4, window_size=3)
    model.train()
    x = torch.randn(1, 6, 64)  # (batch, seq_len, dim)
    out = model(x)
    assert not torch.isnan(out).any(), "Output has NaNs"
    assert out.shape == (1, 6, 64)

def test_inference_kv_cache_rollover():
    model = AttentionWithKVCache(dim=64, num_heads=4, window_size=4, max_seq_len=8)
    model.eval()
    model.reset_cache()

    with torch.no_grad():
        outputs = []
        for i in range(10):  # Feed 10 tokens, triggering cache rolling
            x = torch.randn(1, 1, 64)
            y = model(x, start_pos=i)
            outputs.append(y)
        final_out = torch.cat(outputs, dim=1)

    assert final_out.shape == (1, 10, 64)
    assert not torch.isnan(final_out).any(), "NaNs in output after cache rollover"
    assert model.cache_k.shape == (8, 2, 16), "KV cache size mismatch"

def test_grouped_query_attention_sharing():
    model = AttentionWithKVCache(dim=64, num_heads=8, num_kv_heads=2, window_size=5)
    x = torch.randn(1, 6, 64)
    model.train()
    out = model(x)

    assert out.shape == (1, 6, 64)
    assert model.repeats == 4, "KV head sharing is incorrect"

def test_cache_reset():
    model = AttentionWithKVCache(dim=64, num_heads=4, window_size=4)
    model.eval()
    model.reset_cache()

    x = torch.randn(1, 2, 64)
    _ = model(x)
    assert model.cache_pos > 0

    model.reset_cache()
    assert model.cache_pos == 0
    assert torch.all(model.cache_k == 0)