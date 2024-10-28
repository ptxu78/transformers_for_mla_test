import torch
import time
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention
'''
KV Cache Size = 2 × (batch size × num attention heads × sequence length × head dim × dtype size) 
'''
def calculate_kv_cache_size(batch_size, sequence_length, num_attention_heads, head_dim, dtype_size):
    kv_cache_size_bytes = 2 * (batch_size * num_attention_heads * sequence_length * head_dim * dtype_size)
    kv_cache_size_mb = kv_cache_size_bytes / (1024 ** 3)  
    return kv_cache_size_mb

def run_test(batch_size, input_len, output_len, config, Attention):
    print("="*40)
    print(f"Batch Size: {batch_size}, Input Len: {input_len}, Output Len: {output_len}")
    # 模拟输入张量
    input_ids = torch.randint(0, config.vocab_size, (batch_size, input_len)).cuda()
    attention_mask = torch.ones(batch_size, input_len).cuda()
    position_ids = torch.arange(input_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).cuda()

    embedding_layer = torch.nn.Embedding(config.vocab_size, config.hidden_size).cuda()
    hidden_states = embedding_layer(input_ids)  # Shape: [batch_size, input_len, hidden_size]

    # 模拟过去的 `key` 和 `value` 缓存
    past_key_values = None  # 模拟首次运行时没有缓存
    # 数据类型大小，假设使用 float32
    dtype_size = torch.finfo(torch.float32).bits // 8  
    outputs = Attention(
        hidden_states=hidden_states,
        attention_mask=None,  # Assuming no attention mask is needed here
        position_ids=position_ids,
        past_key_value=None,
        output_attentions=False,
        use_cache=True
    )

    memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # 转为GB
    print(f"Memory Allocated First Time: {memory_allocated:.4f} GB")

    # Update past_key_values
    past_key_values = outputs[1]

    max_kv_cache_size = calculate_kv_cache_size(
        batch_size=batch_size,
        sequence_length=input_len,
        num_attention_heads=config.num_attention_heads,
        head_dim=config.head_dim,  
        dtype_size=dtype_size
    )
    print(f"Theoretical Maximum KVCache Size: {max_kv_cache_size:.4f} GB")

    torch.cuda.synchronize()
    start_time = time.time()

    # 前向传播
    with torch.no_grad():
        for step in range(output_len):
            current_input_id = torch.randint(0, config.vocab_size, (batch_size, 1)).cuda()
            current_hidden_state = embedding_layer(current_input_id)  # Shape: [batch_size, 1, hidden_size]
            position_id = input_len + step
            position_ids = torch.full((batch_size, 1), position_id, dtype=torch.long).cuda()
            outputs = Attention(
                hidden_states=current_hidden_state,
                attention_mask=None,  # Attention mask can be None during generation
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True
            )
            past_key_values = outputs[1]
            # sequence_length = input_len + step + 1  # Total sequence length
            # kv_cache_size = calculate_kv_cache_size(
            #     batch_size, sequence_length, config.num_attention_heads, config.head_dim, dtype_size
            # )
            # if (step + 1) % 100 == 0:
            #     print(f"Step {step + 1}, KV Cache Size: {kv_cache_size:.4f} GB")
            # torch.cuda.empty_cache()

    torch.cuda.synchronize()
    end_time = time.time()

    # 计算延迟和最大 GPU 内存使用量
    elapsed_time = end_time - start_time
    memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # 转为GB

    print(f"Memory Allocated: {memory_allocated:.4f}GB")
    print(f"Elapsed Times: {elapsed_time:.4f}s")
    # 重置最大内存统计
    torch.cuda.reset_max_memory_allocated()

# 测试不同配置
config = LlamaConfig()
Attention = LlamaAttention(config=config, layer_idx=0).cuda()  # 添加 layer_idx

# 测试不同设置
# batch_size, input_len, output_len
run_test(32, 32, 32, config, Attention)
# run_test(32, 32, 128, config, Attention)
# run_test(32, 32, 1024, config, Attention)
# run_test(128, 128, 128, config, Attention)
# run_test(128, 128, 32000, config, Attention)
# run_test(128, 1024, 128, config, Attention)
# run_test(128, 1024, 32000, config, Attention)
# run_test(1024, 128, 128, config, Attention)
# run_test(1024, 128, 32000, config, Attention)
# 测不了
# run_test(1024, 1024, 128, config, Attention) 
# run_test(1024, 1024, 32000, config, Attention)