---
license: apache-2.0
language:
- zh
- en
pipeline_tag: text-generation
library_name: transformers
---
<div align="center">
<img src="https://github.com/OpenBMB/MiniCPM/blob/main/assets/minicpm_logo.png?raw=true" width="500em" ></img> 
</div>

<p align="center">
<a href="https://github.com/OpenBMB/MiniCPM/" target="_blank">MiniCPM Repo</a> |
<a href="https://arxiv.org/abs/2404.06395" target="_blank">MiniCPM Paper</a> |
<a href="https://github.com/OpenBMB/MiniCPM-V/" target="_blank">MiniCPM-V Repo</a> |
Join us in <a href="https://discord.gg/3cGQn9b3YM" target="_blank">Discord</a> and <a href="https://github.com/OpenBMB/MiniCPM/blob/main/assets/wechat.jpg" target="_blank">WeChat</a>
 
</p>

## Introduction
MiniCPM3-4B is the 3rd generation of MiniCPM series. The overall performance of MiniCPM3-4B surpasses Phi-3.5-mini-Instruct and GPT-3.5-Turbo-0125, being comparable with many recent 7B~9B models.

Compared to MiniCPM1.0/MiniCPM2.0, MiniCPM3-4B has a more powerful and versatile skill set to enable more general usage. MiniCPM3-4B supports function call, along with code interpreter. Please refer to [Advanced Features](https://github.com/OpenBMB/MiniCPM/tree/main?tab=readme-ov-file#%E8%BF%9B%E9%98%B6%E5%8A%9F%E8%83%BD) for usage guidelines.

MiniCPM3-4B has a 32k context window. Equipped with LLMxMapReduce, MiniCPM3-4B can handle infinite context theoretically, without requiring huge amount of memory.

## Usage
### Inference with Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

path = "openbmb/MiniCPM3-4B"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

messages = [
    {"role": "user", "content": "推荐5个北京的景点。"},
]
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(device)

model_outputs = model.generate(
    model_inputs,
    max_new_tokens=1024,
    top_p=0.7,
    temperature=0.7
)

output_token_ids = [
    model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs))
]

responses = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
print(responses)
```

### Inference with [vLLM](https://github.com/vllm-project/vllm)

For now, you need to install our forked version of vLLM.

```bash
pip install git+https://github.com/OpenBMB/vllm.git@minicpm3
```

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_name = "openbmb/MiniCPM3-4B"
prompt = [{"role": "user", "content": "推荐5个北京的景点。"}]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

llm = LLM(
    model=model_name,
    trust_remote_code=True,
    tensor_parallel_size=1
)
sampling_params = SamplingParams(top_p=0.7, temperature=0.7, max_tokens=1024, repetition_penalty=1.02)

outputs = llm.generate(prompts=input_text, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)
```

## Evaluation Results

<table>
    <tr>
        <td>Benchmark</td>
        <td>Qwen2-7B-Instruct</td>
        <td>GLM-4-9B-Chat</td>
        <td>Gemma2-9B-it</td>
        <td>Llama3.1-8B-Instruct</td>
        <td>GPT-3.5-Turbo-0125</td>
        <td>Phi-3.5-mini-Instruct(3.8B)</td>
        <td>MiniCPM3-4B </td>
    </tr>
    <tr>
        <td colspan="15" align="left"><strong>English</strong></td>
    </tr>
    <tr>
        <td>MMLU</td>
        <td>70.5</td>
        <td>72.4</td>
        <td>72.6</td>
        <td>69.4</td>
        <td>69.2</td>
        <td>68.4</td>
        <td>67.2 </td>
    </tr>
    <tr>
        <td>BBH</td>
        <td>64.9</td>
        <td>76.3</td>
        <td>65.2</td>
        <td>67.8</td>
        <td>70.3</td>
        <td>68.6</td>
        <td>70.2 </td>
    </tr>
    <tr>
        <td>MT-Bench</td>
        <td>8.41</td>
        <td>8.35</td>
        <td>7.88</td>
        <td>8.28</td>
        <td>8.17</td>
        <td>8.60</td>
        <td>8.41 </td>
    </tr>
    <tr>
        <td>IFEVAL (Prompt Strict-Acc.)</td>
        <td>51.0</td>
        <td>64.5</td>
        <td>71.9</td>
        <td>71.5</td>
        <td>58.8</td>
        <td>49.4</td>
        <td>68.4 </td>
    </tr>
    <tr>
        <td colspan="15" align="left"><strong>Chinese</strong></td>
    </tr>
    <tr>
        <td>CMMLU</td>
        <td>80.9</td>
        <td>71.5</td>
        <td>59.5</td>
        <td>55.8</td>
        <td>54.5</td>
        <td>46.9</td>
        <td>73.3 </td>
    </tr>
    <tr>
        <td>CEVAL</td>
        <td>77.2</td>
        <td>75.6</td>
        <td>56.7</td>
        <td>55.2</td>
        <td>52.8</td>
        <td>46.1</td>
        <td>73.6 </td>
    </tr>
    <tr>
        <td>AlignBench v1.1</td>
        <td>7.10</td>
        <td>6.61</td>
        <td>7.10</td>
        <td>5.68</td>
        <td>5.82</td>
        <td>5.73</td>
        <td>6.74 </td>
    </tr>
    <tr>
        <td>FollowBench-zh (SSR)</td>
        <td>63.0</td>
        <td>56.4</td>
        <td>57.0</td>
        <td>50.6</td>
        <td>64.6</td>
        <td>58.1</td>
        <td>66.8 </td>
    </tr>
    <tr>
        <td colspan="15" align="left"><strong>Math</strong></td>
    </tr>
    <tr>
        <td>MATH</td>
        <td>49.6</td>
        <td>50.6</td>
        <td>46.0</td>
        <td>51.9</td>
        <td>41.8</td>
        <td>46.4</td>
        <td>46.6 </td>
    </tr>
    <tr>
        <td>GSM8K</td>
        <td>82.3</td>
        <td>79.6</td>
        <td>79.7</td>
        <td>84.5</td>
        <td>76.4</td>
        <td>82.7</td>
        <td>81.1 </td>
    </tr>
    <tr>
        <td>MathBench</td>
        <td>63.4</td>
        <td>59.4</td>
        <td>45.8</td>
        <td>54.3</td>
        <td>48.9</td>
        <td>54.9</td>
        <td>65.6 </td>
    </tr>
    <tr>
        <td colspan="15" align="left"><strong>Code</strong></td>
    </tr>
    <tr>
        <td>HumanEval+</td>
        <td>70.1</td>
        <td>67.1</td>
        <td>61.6</td>
        <td>62.8</td>
        <td>66.5</td>
        <td>68.9</td>
        <td>68.3 </td>
    </tr>
    <tr>
        <td>MBPP+</td>
        <td>57.1</td>
        <td>62.2</td>
        <td>64.3</td>
        <td>55.3</td>
        <td>71.4</td>
        <td>55.8</td>
        <td>63.2 </td>
    </tr>
    <tr>
        <td>LiveCodeBench v3</td>
        <td>22.2</td>
        <td>20.2</td>
        <td>19.2</td>
        <td>20.4</td>
        <td>24.0</td>
        <td>19.6</td>
        <td>22.6 </td>
    </tr>
    <tr>
        <td colspan="15" align="left"><strong>Function Call</strong></td>
    </tr>
    <tr>
        <td>BFCL v2</td>
        <td>71.6</td>
        <td>70.1</td>
        <td>19.2</td>
        <td>73.3</td>
        <td>75.4</td>
        <td>48.4</td>
        <td>76.0 </td>
    </tr>
    <tr>
        <td colspan="15" align="left"><strong>Overall</strong></td>
    </tr>
    <tr>
        <td>Average</td>
        <td>65.3</td>
        <td>65.0</td>
        <td>57.9</td>
        <td>60.8</td>
        <td>61.0</td>
        <td>57.2</td>
        <td><strong>66.3</strong></td>
    </tr>
</table>


## Statement
* As a language model, MiniCPM3-4B generates content by learning from a vast amount of text.
* However, it does not possess the ability to comprehend or express personal opinions or value judgments.
* Any content generated by MiniCPM3-4B does not represent the viewpoints or positions of the model developers.
* Therefore, when using content generated by MiniCPM3-4B, users should take full responsibility for evaluating and verifying it on their own.

## LICENSE
* This repository is released under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) License. 
* The usage of MiniCPM3-4B model weights must strictly follow [MiniCPM Model License.md](https://github.com/OpenBMB/MiniCPM/blob/main/MiniCPM%20Model%20License.md).
* The models and weights of MiniCPM3-4B are completely free for academic research. after filling out a ["questionnaire"](https://modelbest.feishu.cn/share/base/form/shrcnpV5ZT9EJ6xYjh3Kx0J6v8g) for registration, are also available for free commercial use.

## Citation

```
@article{hu2024minicpm,
  title={MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies},
  author={Hu, Shengding and Tu, Yuge and Han, Xu and He, Chaoqun and Cui, Ganqu and Long, Xiang and Zheng, Zhi and Fang, Yewei and Huang, Yuxiang and Zhao, Weilin and others},
  journal={arXiv preprint arXiv:2404.06395},
  year={2024}
}
```