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
#include "../../AgentClient.hpp"
#else
#include "../../basic.hpp"
#endif

#include <torch/torch.h>

#define LAYER_INDEX 3

struct ResBImpl : torch::nn::Module {
    std::vector<torch::nn::Linear> layers;
    int num_layers;

    ResBImpl(int hidden_size, int num_layers) : num_layers(num_layers) {
        for (int i = 0; i < num_layers; ++i)
            layers.push_back(
                register_module("lin" + std::to_string(i), torch::nn::Linear(hidden_size, hidden_size))
            );
    }

    torch::Tensor forward(torch::Tensor X) {
        auto y = X.clone();
        auto x = y * y.numel() / (y.abs().sum().detach() + 1e-8);
        for (int i = 0; i < num_layers; ++i) {
            y = torch::relu(layers[i]->forward(x)) + x;
            x = y * y.numel() / (y.abs().sum().detach() + 1e-8);
        }
        return x;
    }

};
TORCH_MODULE(ResB);

struct GameCNNImpl : torch::nn::Module {
    std::vector<torch::nn::Conv2d> conv;
    int channels, d_out, layers, n;

    GameCNNImpl(int channels, int d_out, int layers, int n): channels(channels), d_out(d_out), layers(layers), n(n) {
        for (int i = 0; i < layers; ++i)
            conv.push_back(
                register_module("conv" + std::to_string(i),
                torch::nn::Conv2d(torch::nn::Conv2dOptions((i ? d_out : channels), d_out, 3).stride(2).padding(0).bias(false)))
            );
    }

    torch::Tensor forward(torch::Tensor x) {
        auto y = x.clone();
        for (int i = 0; i < layers; ++i)
            y = conv[i]->forward(y);
        return y;
    }
};
TORCH_MODULE(GameCNN);

struct BackboneImpl : torch::nn::Module {
    GameCNN cnn{nullptr};
    torch::nn::GRU gru0{nullptr}, gru1{nullptr};
    torch::nn::Sequential combined_processor{nullptr};

    int num_channels, grid_x, grid_y, hidden_size, num_actions;
    torch::Tensor action_input, h_state[2];

    BackboneImpl(int num_channels = 32, int grid_x = 31, int grid_y = 31, int hidden_size = 160, int num_actions = 9)
        : num_channels(num_channels), grid_x(grid_x), grid_y(grid_y),
          hidden_size(hidden_size), num_actions(num_actions) {
        cnn = register_module("cnn", GameCNN(num_channels, hidden_size, 4, grid_x));
        gru0 = register_module("gru0", torch::nn::GRU(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(1)));
        combined_processor = register_module("combined_processor", torch::nn::Sequential(
            torch::nn::Linear(2 * hidden_size + num_actions, hidden_size)
        ));
        gru1 = register_module("gru1", torch::nn::GRU(torch::nn::GRUOptions(hidden_size, hidden_size).num_layers(1)));
        reset_memory();
    }

    void reset_memory() {
        action_input = torch::zeros({num_actions});
        action_input[0] += 1;
        h_state[0] = torch::zeros({1, 1, hidden_size});
        h_state[1] = torch::zeros({1, 1, hidden_size});
    }

    void update_actions(torch::Tensor one_hot) {
        action_input = one_hot.clone();
    }

    torch::Tensor forward(const torch::Tensor &x) {
        auto feat = cnn->forward(x);
        feat = feat * hidden_size / (feat.abs().sum().detach() + 1e-8);

        auto r0 = gru0->forward(feat.view({1, 1, -1}), h_state[0]);
        auto out_seq = std::get<0>(r0).view({-1});
        out_seq = out_seq * hidden_size / (out_seq.abs().sum().detach() + 1e-8);
        h_state[0] = std::get<1>(r0);

        std::vector<torch::Tensor> y;
        std::vector<std::vector<int>> d = {{-1, 0}, {0, -1}, {0, 0}, {0, 1}, {1, 0}};
        for (auto e: d)
            for (int j = 0; j < num_channels; ++j)
                y.push_back(
                    x[0][j][grid_x / 2 + e[0]][grid_y / 2 + e[1]].clone()
                );
        auto pov = torch::cat({torch::stack(y).view({-1}), action_input});
        auto combined = torch::cat({out_seq + feat.view({-1}), pov * hidden_size / (pov.abs().sum().detach() + 1e-8)});

        auto gated = combined_processor->forward(combined);
        gated = gated * hidden_size / (gated.abs().sum().detach() + 1e-8);

        auto r1 = gru1->forward(gated.view({1, 1, -1}), h_state[1]);
        auto out = std::get<0>(r1).view({-1});
        out = out * hidden_size / (out.abs().sum().detach() + 1e-8) + gated;
        h_state[1] = std::get<1>(r1);

        return out;
    }
};
TORCH_MODULE(Backbone);

struct AgentModelImpl : torch::nn::Module {
    Backbone backbone{nullptr};
    torch::nn::Sequential value_head{nullptr}, policy_head{nullptr};

    int num_channels, grid_x, grid_y, hidden_size, num_actions;

    AgentModelImpl(int num_channels = 32, int grid_x = 31, int grid_y = 31, int hidden_size = 160, int num_actions = 9)
        : num_channels(num_channels), grid_x(grid_x), grid_y(grid_y), hidden_size(hidden_size), num_actions(num_actions) {
        backbone = register_module("backbone", Backbone(num_channels, grid_x, grid_y, hidden_size, num_actions));
        value_head = register_module("value", torch::nn::Sequential(
            ResB(hidden_size, LAYER_INDEX), torch::nn::Linear(hidden_size, 1)
        ));
        policy_head = register_module("policy", torch::nn::Sequential(
            ResB(hidden_size, LAYER_INDEX), torch::nn::Linear(hidden_size, num_actions)
        ));
        backbone->reset_memory();
    }

    void freeze_backbone(){
        for (auto& p : backbone->parameters())
            p.set_requires_grad(false);
    }

    void reset_memory() {
        backbone->reset_memory();
    }

    void update_actions(torch::Tensor one_hot) {
        backbone->update_actions(one_hot);
    }

    std::vector<torch::Tensor> forward(torch::Tensor x) {
        auto gated = backbone->forward(x);

        auto logits = policy_head->forward(gated).view({-1});
        auto p = torch::softmax(logits, -1) + 1e-8;
        
        auto v = torch::sigmoid(value_head->forward(gated)).view({-1});

        return {p, v};
    }
};
TORCH_MODULE(AgentModel);