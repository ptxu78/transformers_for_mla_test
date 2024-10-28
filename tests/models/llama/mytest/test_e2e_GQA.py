import torch
import time
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import DynamicCache
from transformers import AutoTokenizer

if torch.cuda.is_available():
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
else:
    device = torch.device("cpu")  # 如果没有GPU，就使用CPU
print(f"Using device: {device}")


def run_test(batch_size, input_len, output_len):
    # 模型准备
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", device_map=device, torch_dtype=torch.bfloat16
    )
    # 模型调整
    
    
    # 输入准备 真实
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    # input_text = ["Tell me about the french revolution."] * batch_size
    # model_inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # 输入准备 随机
    model_inputs = dict()
    config = LlamaConfig()
    model_inputs['input_ids'] = torch.randint(0, config.vocab_size, (batch_size, input_len)).cuda()
    model_inputs['attention_mask'] = torch.ones((batch_size, input_len)).cuda()
    
    generated_ids = model.generate(**model_inputs, max_new_tokens=output_len, do_sample=False, attn_implementation = "flash_attention_2")
    print("generated_ids:", generated_ids)
    
# run_test(batch_size=1024, input_len=64, output_len=2048)
run_test(batch_size=128, input_len=128, output_len=2048)