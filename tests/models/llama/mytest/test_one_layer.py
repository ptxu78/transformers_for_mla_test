import torch
import time
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM
from transformers import DynamicCache

def run_test(batch_size, input_len, output_len):
    # 模型配置
    config = LlamaConfig(num_hidden_layers=1)

    # 构建单层 LLaMA 模型
    model = LlamaForCausalLM(config).cuda()

    # 将模型设置为评估模式（推理模式）
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (batch_size, input_len)).cuda()
    embedding_layer = torch.nn.Embedding(config.vocab_size, config.hidden_size).cuda()
    hidden_states = embedding_layer(input_ids)  # Shape: [batch_size, input_len, hidden_size]
    model_param = {}
    model_param["use_cache"] = True
    
    model_param["past_key_values"] = DynamicCache()
    model_param["attention_mask"] = torch.ones(batch_size, input_len).cuda()
    model_param["position_ids"] = torch.arange(input_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).cuda()
    
    with torch.no_grad():
        for step in range(output_len):
            outputs = model(**model_param, input_ids=input_ids)

            new_kv_cache = outputs.past_key_values

            print(f"Step {step}:")
            for layer, (key, value) in enumerate(new_kv_cache):
                print(f"  Layer {layer}: key shape = {key.shape}, value shape = {value.shape}")

            input_ids = torch.randint(0, config.vocab_size, (batch_size, 1)).cuda()
    
run_test(batch_size=8, input_len=32, output_len=16)