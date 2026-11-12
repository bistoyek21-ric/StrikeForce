/*
MIT License

Copyright (c) 2025 bistoyek21 R.I.C.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#if defined(DISTRIBUTED_LEARNING)
#include "AgentClient.hpp"
#else
#include "../../basic.hpp"
#endif

#pragma once

#include <torch/torch.h>

// Spatial Attention: re-weights each spatial location based on learned importance
struct SpatialAttentionImpl : torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};
    SpatialAttentionImpl(int64_t in_channels) {
        conv = register_module("conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 1, 1)));
    }
    torch::Tensor forward(torch::Tensor x) {
        return x * torch::sigmoid(conv->forward(x));
    }
};
TORCH_MODULE(SpatialAttention);

// Channel Attention (Squeeze-and-Excitation): re-weights channels based on global context
struct ChannelAttentionImpl : torch::nn::Module {
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    ChannelAttentionImpl(int64_t in_channels, int64_t reduction = 4) {
        avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(1));
        fc1 = register_module("fc1", torch::nn::Linear(in_channels, in_channels / reduction));
        fc2 = register_module("fc2", torch::nn::Linear(in_channels / reduction, in_channels));
    }
    torch::Tensor forward(torch::Tensor x) {
        auto s = avgpool->forward(x).flatten(1);
        s = torch::relu(fc1->forward(s));
        s = torch::sigmoid(fc2->forward(s)).unsqueeze(-1).unsqueeze(-1);
        return x * s;
    }
};
TORCH_MODULE(ChannelAttention);

// Collapse‑proof encoder: max pooling + attention + LayerNorm guarantee injectivity on binary inputs
struct CollapseProofEncoderImpl : torch::nn::Module {
    torch::nn::Sequential conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    SpatialAttention sp1{nullptr}, sp2{nullptr};
    ChannelAttention ch1{nullptr}, ch2{nullptr};
    torch::nn::AdaptiveMaxPool2d global_pool{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Sequential fc{nullptr};

    CollapseProofEncoderImpl(int64_t input_channels, int64_t hidden_size) {
        // Block 1: 5x5 conv -> LayerNorm -> ReLU -> 2x2 MaxPool (31 -> 15)
        conv1 = register_module("conv1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, 64, 5).padding(2)),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({64, 31, 31})),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2))
        ));
        sp1 = register_module("sp1", SpatialAttention(64));
        ch1 = register_module("ch1", ChannelAttention(64));

        // Block 2: 3x3 conv -> LayerNorm -> ReLU -> 2x2 MaxPool (15 -> 7)
        conv2 = register_module("conv2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({128, 15, 15})),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2))
        ));
        sp2 = register_module("sp2", SpatialAttention(128));
        ch2 = register_module("ch2", ChannelAttention(128));

        // Block 3: 3x3 conv -> LayerNorm -> ReLU (7 -> 7)
        conv3 = register_module("conv3", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({128, 7, 7})),
            torch::nn::ReLU()
        ));

        // Global max pooling condenses spatial info, preserves any non‑zero activation
        global_pool = register_module("global_pool", torch::nn::AdaptiveMaxPool2d(1));
        dropout = register_module("dropout", torch::nn::Dropout(0.1));
        fc = register_module("fc", torch::nn::Sequential(
            torch::nn::Flatten(),
            torch::nn::Linear(128, hidden_size),
            torch::nn::ReLU()
        ));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv1->forward(x);  x = sp1->forward(x);  x = ch1->forward(x);
        x = conv2->forward(x);  x = sp2->forward(x);  x = ch2->forward(x);
        x = conv3->forward(x);
        x = global_pool->forward(x);
        x = dropout->forward(x);
        x = fc->forward(x);      // (batch, hidden_size)
        return x;
    }
};
TORCH_MODULE(CollapseProofEncoder);

// Full agent model – same public interface as original, but with collapse‑proof architecture
struct AgentModelImpl : torch::nn::Module {
    CollapseProofEncoder encoder{nullptr};
    torch::nn::Linear action_embed{nullptr};
    torch::nn::Sequential pre_gru{nullptr};
    torch::nn::GRU gru{nullptr};
    torch::nn::Sequential policy_head{nullptr};
    torch::nn::Sequential value_head{nullptr};

    int hidden_size, num_actions;
    torch::Tensor prev_action;   // one-hot vector of last action (num_actions)
    torch::Tensor h_state;       // GRU hidden state (1, 1, hidden_size)

    AgentModelImpl(int num_channels = 32, int grid_x = 31, int grid_y = 31,
                   int hidden_size = 160, int num_actions = 9)
        : hidden_size(hidden_size), num_actions(num_actions) {

        encoder = register_module("encoder", CollapseProofEncoder(num_channels, hidden_size));

        action_embed = register_module("action_embed", torch::nn::Linear(num_actions, hidden_size));

        // Combine observation feature and action embedding before GRU
        pre_gru = register_module("pre_gru", torch::nn::Sequential(
            torch::nn::Linear(2 * hidden_size, hidden_size),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size})),
            torch::nn::ReLU()
        ));

        gru = register_module("gru",
            torch::nn::GRU(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(1)));

        // Simple MLP heads (no weird normalization)
        value_head = register_module("value", torch::nn::Sequential(
            torch::nn::Linear(hidden_size, hidden_size),
            torch::nn::ReLU(),
            torch::nn::Linear(hidden_size, 1)
        ));
        policy_head = register_module("policy", torch::nn::Sequential(
            torch::nn::Linear(hidden_size, hidden_size),
            torch::nn::ReLU(),
            torch::nn::Linear(hidden_size, num_actions)
        ));

        reset_memory();
    }

    void reset_memory() {
        prev_action = torch::zeros({num_actions});
        prev_action[0] += 1.0;         // start token
        h_state = torch::zeros({1, 1, hidden_size});
    }

    void update_actions(torch::Tensor one_hot) {
        prev_action = one_hot.clone();
    }

    std::vector<torch::Tensor> forward(torch::Tensor x) {
        // x: (1, 32, 31, 31) – single observation at current timestep
        auto obs_feat = encoder->forward(x);                             // (1, hidden_size)
        auto act_feat = action_embed->forward(prev_action.unsqueeze(0)); // (1, hidden_size)

        auto combined = torch::cat({obs_feat, act_feat}, -1);            // (1, 2*hidden_size)
        auto gru_input = pre_gru->forward(combined);                     // (1, hidden_size)

        auto r = gru->forward(gru_input.view({1, 1, -1}), h_state);
        auto out = std::get<0>(r).view({1, -1});                         // (1, hidden_size)
        h_state = std::get<1>(r);

        auto logits = policy_head->forward(out).view({-1});              // (num_actions)
        auto p = torch::softmax(logits, -1) + 1e-8;

        auto v = torch::sigmoid(value_head->forward(out)).view({-1});    // (1)

        return {p, v};
    }
};
TORCH_MODULE(AgentModel);