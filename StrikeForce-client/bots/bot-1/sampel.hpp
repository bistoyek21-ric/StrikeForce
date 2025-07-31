#pragma once
#include <torch/torch.h>
#include <vector>

// Squeeze-and-Excitation block
struct SEBlockImpl : torch::nn::Module {
    int channels;
    torch::nn::AdaptiveAvgPool2d avg_pool{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    SEBlockImpl(int c, int reduction = 16)
      : channels(c) {
        avg_pool = register_module("se_avg_pool", torch::nn::AdaptiveAvgPool2d(1));
        fc1 = register_module("se_fc1", torch::nn::Linear(c, c / reduction));
        fc2 = register_module("se_fc2", torch::nn::Linear(c / reduction, c));
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: [N, C, H, W]
        auto y = avg_pool->forward(x).view({x.size(0), channels});        
        y = torch::relu(fc1->forward(y));
        y = torch::sigmoid(fc2->forward(y));
        y = y.view({x.size(0), channels, 1, 1});
        return x * y;
    }
};
TORCH_MODULE(SEBlock);

// Residual block with SE
struct ResidualSEImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    SEBlock se{nullptr};

    ResidualSEImpl(int channels)
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false)));
        bn1   = register_module("bn1",   torch::nn::BatchNorm2d(channels));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 3).padding(1).bias(false)));
        bn2   = register_module("bn2",   torch::nn::BatchNorm2d(channels));
        se    = register_module("se",    SEBlock(channels));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto residual = x;
        x = torch::relu(bn1->forward(conv1->forward(x)));
        x = bn2->forward(conv2->forward(x));
        x = se->forward(x);
        return torch::relu(x + residual);
    }
};
TORCH_MODULE(ResidualSE);

// LC0 Neural Network
struct Lc0NetImpl : torch::nn::Module {
    // constants
    static constexpr int INPUT_PLANES = 112;
    static constexpr int BOARD_SIZE = 8;
    int filters;
    int blocks;

    // modules
    torch::nn::Conv2d stem_conv{nullptr};
    torch::nn::BatchNorm2d stem_bn{nullptr};
    std::vector<ResidualSE> residuals;
    torch::nn::Conv2d policy_conv1{nullptr}, policy_conv2{nullptr};
    torch::nn::Conv2d value_conv1{nullptr}, value_conv2{nullptr};
    torch::nn::Linear value_fc{nullptr};

    Lc0NetImpl(int filters_ = 256, int blocks_ = 20)
      : filters(filters_), blocks(blocks_) {
        // Stem
        stem_conv = register_module("stem_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(INPUT_PLANES, filters, 3).padding(1).bias(false)));
        stem_bn   = register_module("stem_bn",   torch::nn::BatchNorm2d(filters));
        
        // Residual Tower
        for (int i = 0; i < blocks; ++i) {
            auto block = ResidualSE(filters);
            residuals.push_back(register_module("residual_" + std::to_string(i), block));
        }

        // Policy Head
        policy_conv1 = register_module("policy_conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, filters, 3).padding(1).bias(false)));
        policy_conv2 = register_module("policy_conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, 73, 1)));

        // Value Head
        value_conv1 = register_module("value_conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(filters, 32, 1)));
        value_conv2 = register_module("value_conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 128, 1)));
        value_fc    = register_module("value_fc",    torch::nn::Linear(128 * BOARD_SIZE * BOARD_SIZE, 3));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        // x: [N, 112, 8, 8]
        x = torch::relu(stem_bn->forward(stem_conv->forward(x)));
        for (auto& block : residuals) x = block->forward(x);
        
        // Policy
        auto p = torch::relu(policy_conv1->forward(x));
        p = policy_conv2->forward(p);               // [N, 73, 8, 8]
        p = p.view({p.size(0), -1});                // flatten to [N, 73*8*8]

        // Value
        auto v = torch::relu(value_conv1->forward(x));
        v = torch::relu(value_conv2->forward(v));   // [N, 128, 8, 8]
        v = v.view({v.size(0), -1});                // flatten
        v = value_fc->forward(v);                   // [N, 3]
        v = torch::softmax(v, /*dim=*/1);           // probabilities W/D/L

        return {p, v};
    }
};
TORCH_MODULE(Lc0Net);

// Agent wrapper for inference
class Lc0Agent {
public:
    Lc0Agent(const std::string& model_path, torch::Device device = torch::kCPU) {
        net = Lc0Net();
        net->to(device);
        torch::load(net, model_path);
        net->eval();
    }

    // Input: tensor [1,112,8,8]
    std::pair<torch::Tensor, torch::Tensor> evaluate(torch::Tensor input) {
        // returns policy logits and W/D/L probabilities
        return net->forward(input);
    }

private:
    Lc0Net net;
};