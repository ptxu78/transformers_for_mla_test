# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the PyTorch LLaMA model."""

import gc
import tempfile
import unittest

import pytest
from packaging import version
from parameterized import parameterized

from transformers import AutoTokenizer, LlamaConfig, StaticCache, is_torch_available, set_seed
from transformers.testing_utils import (
    backend_empty_cache,
    require_bitsandbytes,
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        LlamaForCausalLM,
        LlamaForQuestionAnswering,
        LlamaForSequenceClassification,
        LlamaForTokenClassification,
        LlamaModel,
        LlamaTokenizer,
        LlamaConfig
    )
    from transformers.models.llama.modeling_llama import (
        LlamaLinearScalingRotaryEmbedding, 
        LlamaRotaryEmbedding, 
        LlamaAttention
    )

# class LlamaAttentionTester(unittest.TestCase):
#     def setUp(self):
#         # 配置单层Llama Attention的超参数
#         self.batch_size = 2
#         self.seq_length = 10
#         self.hidden_size = 32
#         self.num_attention_heads = 4
#         self.head_dim = 8
#         self.config = LlamaConfig(
#             hidden_size=self.hidden_size,
#             num_attention_heads=self.num_attention_heads,
#             head_dim=self.head_dim,
#             attention_dropout=0.1,
#             rope_theta=10000,
#             num_key_value_heads=2
#         )
#         self.attention_layer = LlamaAttention(config=self.config, layer_idx=0).cuda()

#     def test_attention_forward(self):
#         # 准备输入数据
#         hidden_states = torch.randn(self.batch_size, self.seq_length, self.hidden_size).cuda()
#         attention_mask = torch.ones(self.batch_size, 1, 1, self.seq_length).cuda()

#         # 测试前向传播
#         outputs, attn_weights, _ = self.attention_layer(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             output_attentions=True
#         )
        
#         # 验证输出形状是否正确
#         self.assertEqual(outputs.shape, (self.batch_size, self.seq_length, self.hidden_size))
#         self.assertEqual(attn_weights.shape, (self.batch_size, self.num_attention_heads, self.seq_length, self.seq_length))

#     def test_attention_with_cache(self):
#         # 模拟带有缓存的输入
#         hidden_states = torch.randn(self.batch_size, 1, self.hidden_size).cuda()
#         attention_mask = torch.ones(self.batch_size, 1, 1, 1).cuda()
        
#         # 初始化前向传播
#         outputs, attn_weights, past_key_value = self.attention_layer(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             use_cache=True
#         )

#         # 确认缓存是否正确生成
#         self.assertIsNotNone(past_key_value)

#         # 再次前向传播，使用缓存
#         new_hidden_states = torch.randn(self.batch_size, 1, self.hidden_size).cuda()
#         new_attention_mask = torch.ones(self.batch_size, 1, 1, 2).cuda()

#         outputs, attn_weights, _ = self.attention_layer(
#             hidden_states=new_hidden_states,
#             attention_mask=new_attention_mask,
#             use_cache=True,
#             past_key_value=past_key_value
#         )

#         # 验证输出形状
#         self.assertEqual(outputs.shape, (self.batch_size, 1, self.hidden_size))

#     def test_attention_no_position_embeddings(self):
#         # 测试不提供位置嵌入的情况
#         hidden_states = torch.randn(self.batch_size, self.seq_length, self.hidden_size).cuda()
#         attention_mask = torch.ones(self.batch_size, 1, 1, self.seq_length).cuda()

#         outputs, attn_weights, _ = self.attention_layer(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_embeddings=None  # 不传入位置嵌入
#         )
        
#         # 验证输出形状是否正确
#         self.assertEqual(outputs.shape, (self.batch_size, self.seq_length, self.hidden_size))

class LlamaAttentionTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        hidden_size=32,
        num_attention_heads=4,
        attention_dropout=0.1,
        max_position_embeddings=512,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings

    def get_config(self):
        return LlamaConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
        )

    def create_and_check_attention(
        self, config, input_ids, attention_mask, position_ids, past_key_values
    ):
        attention_layer = LlamaAttention(config=config, layer_idx=0).cuda()
        attention_layer.eval()

        with torch.no_grad():
            output, attn_weights, _ = attention_layer(
                hidden_states=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
            )

        # 检查输出形状是否正确
        self.parent.assertEqual(
            output.shape,
            (self.batch_size, self.seq_length, self.hidden_size)
        )

    def prepare_inputs_for_test(self):
        config = self.get_config()
        input_ids = torch.randn(self.batch_size, self.seq_length, self.hidden_size).cuda()
        attention_mask = torch.ones(self.batch_size, 1, 1, self.seq_length).cuda()
        position_ids = torch.arange(self.seq_length, dtype=torch.long).unsqueeze(0).expand(self.batch_size, -1).cuda()
        past_key_values = None

        return config, input_ids, attention_mask, position_ids, past_key_values


class LlamaAttentionTest(unittest.TestCase):
    def setUp(self):
        self.attention_tester = LlamaAttentionTester(self)

    def test_attention_layer(self):
        config, input_ids, attention_mask, position_ids, past_key_values = self.attention_tester.prepare_inputs_for_test()
        # print(f"Input IDs shape: {input_ids.shape}")
        self.attention_tester.create_and_check_attention(config, input_ids, attention_mask, position_ids, past_key_values)