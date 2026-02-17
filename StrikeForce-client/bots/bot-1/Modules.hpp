/*
MIT License (c) 2025 bistoyek21 R.I.C.

Modules.hpp  — bot-1
Improvements over bot-1-redesigned:
  • Multi-scale CNN (fine/mid/coarse) with CBAM attention
  • Hierarchical GRU (fast @ every step, slow @ every K steps)
  • Auxiliary heads: inverse dynamics + latent forward model
  • Shared backbone used by AgentModel and AIRL reward net
*/
#pragma once
#if defined(DISTRIBUTED_LEARNING)
#include "../../AgentClient.hpp"
#else
#include "../../basic.hpp"
#endif
#include <torch/torch.h>

// ─────────────────────────────────────────────
//  Hyper-constants
// ─────────────────────────────────────────────
#define LAYER_INDEX    3   // ResB depth
#define SLOW_K         4   // Slow GRU update frequency
#define CBAM_REDUCTION 8   // Channel-attention squeeze ratio

// ─────────────────────────────────────────────
//  ResB  (unchanged — stable primitive)
// ─────────────────────────────────────────────
struct ResBImpl : torch::nn::Module {
    std::vector<torch::nn::Linear> layers;
    int num_layers;

    ResBImpl(int hidden_size, int num_layers) : num_layers(num_layers) {
        for (int i = 0; i < num_layers; ++i)
            layers.push_back(register_module(
                "lin" + std::to_string(i),
                torch::nn::Linear(hidden_size, hidden_size)));
    }

    torch::Tensor forward(torch::Tensor x) {
        for (int i = 0; i < num_layers; ++i)
            x = torch::relu(layers[i]->forward(x)) + x;
        return x;
    }
};
TORCH_MODULE(ResB);

// ─────────────────────────────────────────────
//  CBAM  (Convolutional Block Attention Module)
//  Applied to spatial feature maps before pooling.
//  Channel attention tells the network WHICH
//  feature types matter (walls / enemies / bullets).
//  Spatial attention tells it WHERE to look.
// ─────────────────────────────────────────────
struct CBAMImpl : torch::nn::Module {
    torch::nn::Linear  ch_fc1{nullptr}, ch_fc2{nullptr};
    torch::nn::Conv2d  sp_conv{nullptr};
    int C;

    CBAMImpl(int channels) : C(channels) {
        int r = std::max(1, channels / CBAM_REDUCTION);
        ch_fc1   = register_module("ch_fc1", torch::nn::Linear(C, r));
        ch_fc2   = register_module("ch_fc2", torch::nn::Linear(r, C));
        // kernel=7, padding=3 preserves spatial size
        sp_conv  = register_module("sp_conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(2, 1, 7).padding(3).bias(false)));
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: [B, C, H, W]

        // ── Channel attention ──────────────────────
        auto avg_c = x.mean({2, 3});                       // [B, C]
        auto max_c = std::get<0>(x.flatten(2).max(2));     // [B, C]
        auto ch_gate = torch::sigmoid(
            ch_fc2->forward(torch::relu(ch_fc1->forward(avg_c))) +
            ch_fc2->forward(torch::relu(ch_fc1->forward(max_c)))
        );                                                 // [B, C]
        x = x * ch_gate.unsqueeze(-1).unsqueeze(-1);

        // ── Spatial attention ──────────────────────
        auto avg_s = x.mean(1, /*keepdim=*/true);          // [B, 1, H, W]
        auto max_s = std::get<0>(x.max(1, /*keepdim=*/true));
        auto sp_gate = torch::sigmoid(
            sp_conv->forward(torch::cat({avg_s, max_s}, 1)));  // [B, 1, H, W]
        return x * sp_gate;
    }
};
TORCH_MODULE(CBAM);

// ─────────────────────────────────────────────
//  MultiScaleCNN
//  Three parallel branches at different receptive
//  fields.  CBAM is applied to mid and coarse
//  branches (they retain spatial structure before
//  pooling, making spatial attention meaningful).
//
//  Input : [1, 32, 31, 31]
//  Fine  : 4×Conv(stride=2)         → [1, H, 1, 1]
//  Mid   : 2×Conv(stride=2) + CBAM  → pool → [1, H, 1, 1]
//  Coarse: 1×Conv(stride=4) + CBAM  → pool → [1, H, 1, 1]
//  Output: linear([3H]) → [H]
// ─────────────────────────────────────────────
struct MultiScaleCNNImpl : torch::nn::Module {
    // Fine branch — 4 conv layers
    std::vector<torch::nn::Conv2d> fine_convs;
    // Mid branch  — 2 conv layers
    std::vector<torch::nn::Conv2d> mid_convs;
    CBAM mid_cbam{nullptr};
    // Coarse branch — 1 conv layer
    torch::nn::Conv2d coarse_conv{nullptr};
    CBAM coarse_cbam{nullptr};
    // Fusion
    torch::nn::Linear fusion{nullptr};

    int in_channels, H;

    MultiScaleCNNImpl(int in_channels = 32, int H = 160)
        : in_channels(in_channels), H(H) {

        // Fine: 4 × Conv(kernel=3, stride=2, bias=false)
        //   31→15→7→3→1
        for (int i = 0; i < 4; ++i)
            fine_convs.push_back(register_module(
                "fine_c" + std::to_string(i),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(
                    i == 0 ? in_channels : H, H, 3)
                    .stride(2).padding(0).bias(false))));

        // Mid: 2 × Conv(kernel=3, stride=2) + CBAM
        //   31→15→7
        for (int i = 0; i < 2; ++i)
            mid_convs.push_back(register_module(
                "mid_c" + std::to_string(i),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(
                    i == 0 ? in_channels : H, H, 3)
                    .stride(2).padding(0).bias(false))));
        mid_cbam = register_module("mid_cbam", CBAM(H));

        // Coarse: Conv(kernel=3, stride=4) + CBAM
        //   31→8
        coarse_conv = register_module("coarse_c", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, H, 3)
                .stride(4).padding(0).bias(false)));
        coarse_cbam = register_module("coarse_cbam", CBAM(H));

        fusion = register_module("fusion", torch::nn::Linear(3 * H, H));
    }

    torch::Tensor forward(torch::Tensor x) {
        // ── Fine branch ──────────────────────────
        auto f = x;
        for (auto& conv : fine_convs)
            f = torch::relu(conv->forward(f));             // [1, H, 1, 1]
        auto feat_fine = f.view({-1});                     // [H]

        // ── Mid branch ───────────────────────────
        auto m = x;
        for (auto& conv : mid_convs)
            m = torch::relu(conv->forward(m));             // [1, H, 7, 7]
        m = mid_cbam->forward(m);                          // attended
        auto feat_mid = torch::adaptive_avg_pool2d(m, {1, 1}).view({-1}); // [H]

        // ── Coarse branch ────────────────────────
        auto c = torch::relu(coarse_conv->forward(x));     // [1, H, 8, 8]
        c = coarse_cbam->forward(c);
        auto feat_coarse = torch::adaptive_avg_pool2d(c, {1, 1}).view({-1}); // [H]

        // ── Fusion ───────────────────────────────
        auto concat = torch::cat({feat_fine, feat_mid, feat_coarse}); // [3H]
        return torch::relu(fusion->forward(concat));       // [H]
    }
};
TORCH_MODULE(MultiScaleCNN);

// ─────────────────────────────────────────────
//  HierarchicalGRU
//  Fast GRU   : runs every step, hidden = H/2.
//  Slow GRU   : runs every SLOW_K steps on mean-
//               pooled fast states, hidden = H/2.
//  Output     : LayerNorm( linear( [fast || slow] ) )
//               shape [H]
// ─────────────────────────────────────────────
struct HierarchicalGRUImpl : torch::nn::Module {
    torch::nn::GRU      fast_gru{nullptr}, slow_gru{nullptr};
    torch::nn::Linear   fusion{nullptr};
    torch::nn::LayerNorm ln{nullptr};

    int H, H2;
    int step_count = 0;
    torch::Tensor h_fast, h_slow;
    std::vector<torch::Tensor> fast_buffer;

    HierarchicalGRUImpl(int H = 160) : H(H), H2(H / 2) {
        fast_gru = register_module("fast_gru",
            torch::nn::GRU(torch::nn::GRUOptions(H, H2).num_layers(1)));
        slow_gru = register_module("slow_gru",
            torch::nn::GRU(torch::nn::GRUOptions(H2, H2).num_layers(1)));
        fusion   = register_module("fusion", torch::nn::Linear(H2 + H2, H));
        ln       = register_module("ln", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({H})));
        reset_memory();
    }

    void reset_memory() {
        h_fast = torch::zeros({1, 1, H2});
        h_slow = torch::zeros({1, 1, H2});
        fast_buffer.clear();
        step_count = 0;
    }

    // Returns [H] feature vector
    torch::Tensor forward(torch::Tensor spatial_feat) {
        // spatial_feat: [H]
        auto inp = spatial_feat.view({1, 1, -1});

        // ── Fast GRU ─────────────────────────────
        auto [out_fast_seq, h_fast_new] = fast_gru->forward(inp, h_fast);
        h_fast = h_fast_new;
        auto out_fast = out_fast_seq.view({-1});           // [H2]
        fast_buffer.push_back(out_fast.detach().clone());

        // ── Slow GRU (every SLOW_K fast steps) ───
        if ((int)fast_buffer.size() >= SLOW_K) {
            auto stacked = torch::stack(fast_buffer);      // [K, H2]
            auto summary = stacked.mean(0).view({1, 1, -1}); // [1, 1, H2]
            auto [out_slow_seq, h_slow_new] = slow_gru->forward(summary, h_slow);
            h_slow = h_slow_new;
            fast_buffer.clear();
        }
        auto out_slow = h_slow.view({-1});                 // [H2]

        ++step_count;

        // ── Fuse + LayerNorm ─────────────────────
        auto fused = fusion->forward(torch::cat({out_fast, out_slow})); // [H]
        return ln->forward(fused);
    }
};
TORCH_MODULE(HierarchicalGRU);

// ─────────────────────────────────────────────
//  Backbone  (CNN → HierGRU)
//  Encodes (s_t, action_history) → z_t ∈ R^H
// ─────────────────────────────────────────────
struct BackboneImpl : torch::nn::Module {
    MultiScaleCNN   cnn{nullptr};
    torch::nn::Linear action_embed{nullptr};
    torch::nn::Linear pre_gru{nullptr};    // merge CNN + action before GRU
    HierarchicalGRU hier_gru{nullptr};

    int num_channels, grid_x, grid_y, H, num_actions;
    torch::Tensor last_action;

    BackboneImpl(int num_channels = 32, int grid_x = 31, int grid_y = 31,
                 int H = 160, int num_actions = 9)
        : num_channels(num_channels), grid_x(grid_x), grid_y(grid_y),
          H(H), num_actions(num_actions) {

        cnn          = register_module("cnn",   MultiScaleCNN(num_channels, H));
        action_embed = register_module("aemb",  torch::nn::Linear(num_actions, H / 4));
        pre_gru      = register_module("pre_gru",
                            torch::nn::Linear(H + H / 4, H));
        hier_gru     = register_module("hier_gru", HierarchicalGRU(H));
        reset_memory();
    }

    void reset_memory() {
        last_action = torch::zeros({num_actions});
        last_action[0] = 1.0f;                   // no-op default
        hier_gru->reset_memory();
    }

    void update_action_history(torch::Tensor one_hot) {
        last_action = one_hot.detach().clone();
    }

    // Returns z_t of shape [H]
    // Also returns spatial_feat [H] (for auxiliary tasks)
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        // x: [1, num_channels, grid_x, grid_y]

        // Spatial features
        auto spatial = cnn->forward(x);                      // [H]

        // Action context
        auto action_ctx = torch::relu(
            action_embed->forward(last_action));             // [H/4]

        // Merge before GRU
        auto merged = torch::relu(
            pre_gru->forward(torch::cat({spatial, action_ctx}))); // [H]

        // Hierarchical temporal integration
        auto z = hier_gru->forward(merged);                  // [H]

        return {z, spatial};                                 // (backbone_out, spatial_feat)
    }
};
TORCH_MODULE(Backbone);

// ─────────────────────────────────────────────
//  Auxiliary Heads
//  Operate on consecutive backbone features
//  (z_t, z_{t+1}) to provide extra gradient
//  signal to the shared backbone.
//
//  Task 1 — Inverse dynamics:
//    MLP(z_t || z_{t+1}) → â_t   (9-class CE)
//    Forces backbone to encode action-discriminative
//    information; critical for directional reasoning.
//
//  Task 2 — Latent forward model:
//    MLP(z_t || one_hot(a_t)) → ẑ_{t+1}
//    Loss: cosine similarity with z_{t+1}.detach()
//    Encourages the backbone to model consequences.
// ─────────────────────────────────────────────
struct AuxHeadsImpl : torch::nn::Module {
    torch::nn::Sequential inv_dyn{nullptr};
    torch::nn::Sequential fwd_dyn{nullptr};
    int H, num_actions;

    AuxHeadsImpl(int H = 160, int num_actions = 9)
        : H(H), num_actions(num_actions) {

        // Inverse dynamics: [2H] → [H] → [num_actions]
        inv_dyn = register_module("inv_dyn", torch::nn::Sequential(
            torch::nn::Linear(2 * H, H),
            torch::nn::ReLU(),
            torch::nn::Linear(H, num_actions)
        ));

        // Forward dynamics: [H + num_actions] → [H] → [H]
        fwd_dyn = register_module("fwd_dyn", torch::nn::Sequential(
            torch::nn::Linear(H + num_actions, H),
            torch::nn::ReLU(),
            torch::nn::Linear(H, H)
        ));
    }

    // Inverse dynamics logits
    torch::Tensor inverse(torch::Tensor z_t, torch::Tensor z_t1) {
        return inv_dyn->forward(torch::cat({z_t, z_t1}));   // [num_actions]
    }

    // Predicted next latent
    torch::Tensor forward_pred(torch::Tensor z_t, torch::Tensor one_hot_a) {
        return fwd_dyn->forward(torch::cat({z_t, one_hot_a})); // [H]
    }
};
TORCH_MODULE(AuxHeads);

// ─────────────────────────────────────────────
//  AIRL networks (r_θ and h_φ)
//  Both share the same backbone class but are
//  instantiated separately and do NOT share weights
//  with the policy network (cleaner separation;
//  see note below about potential sharing).
// ─────────────────────────────────────────────
struct AIRLNetsImpl : torch::nn::Module {
    Backbone  backbone{nullptr};
    // r_θ(s,a): state-action reward
    torch::nn::Sequential reward_head{nullptr};
    // h_φ(s): potential / shaping function
    torch::nn::Sequential shaping_head{nullptr};

    int num_channels, grid_x, grid_y, H, num_actions;

    AIRLNetsImpl(int num_channels = 32, int grid_x = 31, int grid_y = 31,
                 int H = 160, int num_actions = 9)
        : num_channels(num_channels), grid_x(grid_x), grid_y(grid_y),
          H(H), num_actions(num_actions) {

        backbone = register_module("backbone",
            Backbone(num_channels, grid_x, grid_y, H, num_actions));

        // r_θ(s,a): scalar reward
        reward_head = register_module("reward", torch::nn::Sequential(
            ResB(H, LAYER_INDEX),
            torch::nn::Linear(H, 1)
        ));

        // h_φ(s): scalar potential
        shaping_head = register_module("shaping", torch::nn::Sequential(
            ResB(H, LAYER_INDEX),
            torch::nn::Linear(H, 1)
        ));

        reset_memory();
    }

    void reset_memory() { backbone->reset_memory(); }

    void update_action_history(torch::Tensor one_hot) {
        backbone->update_action_history(one_hot);
    }

    // Compute r_θ(s,a) and h_φ(s) for one step
    // Returns {r, h}  both scalars
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        auto [z, _] = backbone->forward(x);
        auto r = reward_head->forward(z).squeeze();   // 0-dim scalar
        auto h = shaping_head->forward(z).squeeze();  // 0-dim scalar
        return {r, h};
    }
};
TORCH_MODULE(AIRLNets);

// ─────────────────────────────────────────────
//  AgentModel  (Policy + Value)
//  backbone → z
//  policy_head : z → π(a|s)   [num_actions]
//  value_head  : z → V(s)     scalar (UNBOUNDED)
//  aux_heads   : (z_t, z_t1, a) → aux losses
// ─────────────────────────────────────────────
struct AgentModelImpl : torch::nn::Module {
    Backbone backbone{nullptr};
    torch::nn::Sequential policy_head{nullptr};
    torch::nn::Sequential value_head{nullptr};
    AuxHeads aux_heads{nullptr};

    int num_channels, grid_x, grid_y, H, num_actions;

    AgentModelImpl(int num_channels = 32, int grid_x = 31, int grid_y = 31,
                   int H = 160, int num_actions = 9)
        : num_channels(num_channels), grid_x(grid_x), grid_y(grid_y),
          H(H), num_actions(num_actions) {

        backbone    = register_module("backbone",
            Backbone(num_channels, grid_x, grid_y, H, num_actions));

        policy_head = register_module("policy", torch::nn::Sequential(
            ResB(H, LAYER_INDEX),
            torch::nn::Linear(H, num_actions)               // logits, no softmax here
        ));

        value_head  = register_module("value", torch::nn::Sequential(
            ResB(H, LAYER_INDEX),
            torch::nn::Linear(H, 1)                         // unbounded V(s)
        ));

        aux_heads   = register_module("aux", AuxHeads(H, num_actions));

        reset_memory();
    }

    // Transfer backbone weights from another module (e.g. AIRL reward net)
    void import_backbone_from(const torch::nn::Module& src) {
        for (auto& pair : src.named_parameters()) {
            auto tgt_params = backbone->named_parameters();
            if (tgt_params.contains(pair.key()))
                tgt_params[pair.key()].data().copy_(pair.value().data());
        }
    }

    void freeze_backbone() {
        for (auto& p : backbone->parameters())
            p.set_requires_grad(false);
    }

    void reset_memory() { backbone->reset_memory(); }

    void update_action_history(torch::Tensor one_hot) {
        backbone->update_action_history(one_hot);
    }

    // Returns {probs[num_actions], value 0-dim scalar, z[H]}
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    forward(torch::Tensor x) {
        auto [z, _spatial] = backbone->forward(x);
        auto logits = policy_head->forward(z).view({-1});
        auto probs  = torch::softmax(logits, -1);             // [num_actions]
        // Linear(H,1) on unbatched [H] → [1]; squeeze() → genuine 0-dim scalar
        auto value  = value_head->forward(z).squeeze();       // 0-dim, unbounded
        return {probs, value, z};
    }

    // Auxiliary losses — call once per training cycle after PPO
    // Returns scalar loss (backpropagates through backbone)
    torch::Tensor aux_loss(
        const std::vector<torch::Tensor>& zs,   // z_t for all t
        const std::vector<int>& acts             // actions taken
    ) {
        int T = (int)zs.size() - 1;             // T pairs
        if (T <= 0) return torch::zeros({1});

        auto inv_loss = torch::zeros({1});
        auto fwd_loss = torch::zeros({1});

        for (int t = 0; t < T; ++t) {
            // ── Inverse dynamics (CE) ─────────────
            auto logits = aux_heads->inverse(zs[t], zs[t + 1]);
            inv_loss += torch::nn::functional::cross_entropy(
                logits.unsqueeze(0),                         // [1, num_actions]
                torch::tensor({acts[t]}, torch::kLong));     // [1]

            // ── Forward dynamics (cosine sim) ─────
            auto a_oh = torch::zeros({num_actions});
            a_oh[acts[t]] = 1.0f;
            auto z_pred = aux_heads->forward_pred(zs[t], a_oh);
            // Cosine similarity loss: 1 - cos(z_pred, z_{t+1})
            auto z_tgt  = zs[t + 1].detach();               // stop gradient
            fwd_loss += 1.0f - torch::cosine_similarity(
                z_pred.unsqueeze(0), z_tgt.unsqueeze(0)).squeeze();
        }

        return (inv_loss + fwd_loss) / (float)T;
    }
};
TORCH_MODULE(AgentModel);