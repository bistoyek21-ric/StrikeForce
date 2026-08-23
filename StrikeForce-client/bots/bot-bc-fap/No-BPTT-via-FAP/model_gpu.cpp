/*
MIT License

Copyright (c) 2026 bistoyek21 R.I.C.

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

#include <torch/torch.h>
#include <torch/optim/adamw.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <random>
#include <algorithm>
#include <numeric>
#include <set>
#include <map>
#include <chrono>
#include <iomanip>

//g++ -std=c++17 model_gpu.cpp -o app_gpu -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda && clear && ./app_gpu

namespace fs = std::filesystem;

static constexpr int64_t N_ACTIONS   = 7;

struct PreLNAttnBlockImpl : torch::nn::Module {
    torch::nn::MultiheadAttention attn{nullptr};
    torch::nn::LayerNorm norm_q{nullptr}, norm_kv{nullptr}, norm_ffn{nullptr};
    torch::nn::Linear ffn1{nullptr}, ffn2{nullptr};

    PreLNAttnBlockImpl(int64_t d_model, int64_t n_heads, int64_t ffn_mult = 4) {
        attn = register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(d_model, n_heads)));
        norm_q   = register_module("norm_q",   torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        norm_kv  = register_module("norm_kv",  torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        norm_ffn = register_module("norm_ffn", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        ffn1 = register_module("ffn1", torch::nn::Linear(d_model, ffn_mult * d_model));
        ffn2 = register_module("ffn2", torch::nn::Linear(ffn_mult * d_model, d_model));
    }

    torch::Tensor forward_self(torch::Tensor x) {
        auto normed = norm_q->forward(x);
        auto normed_t = normed.transpose(0, 1);
        auto attn_out = std::get<0>(attn->forward(normed_t, normed_t, normed_t)).transpose(0, 1);
        x = x + attn_out;
        auto ffn_out = ffn2->forward(torch::relu(ffn1->forward(norm_ffn->forward(x))));
        return x + ffn_out;
    }

    torch::Tensor forward_cross(torch::Tensor q, torch::Tensor kv) {
        auto q_normed = norm_q->forward(q);
        auto kv_normed = norm_kv->forward(kv);
        auto q_t  = q_normed.transpose(0, 1);
        auto kv_t = kv_normed.transpose(0, 1);
        auto attn_out = std::get<0>(attn->forward(q_t, kv_t, kv_t)).transpose(0, 1);
        auto x = q + attn_out;
        auto ffn_out = ffn2->forward(torch::relu(ffn1->forward(norm_ffn->forward(x))));
        return x + ffn_out;
    }
};
TORCH_MODULE(PreLNAttnBlock);

struct AFCStageSparseImpl : torch::nn::Module {
    int64_t K, d_tok, d_out, n, n_heads;

    torch::Tensor pos1d, queries, query_pe;
    PreLNAttnBlock axial_block{nullptr};

    torch::Tensor type_embed, token_idx_embed, row_coord_embed, col_coord_embed;
    PreLNAttnBlock fusion_block{nullptr};

    torch::nn::Linear out_proj{nullptr}, proj{nullptr};

    AFCStageSparseImpl(int64_t in_channels, int64_t d_tok_, int64_t d_out_,
                       int64_t n_, int64_t K_ = 6, int64_t n_heads_ = 4)
        : K(K_), d_tok(d_tok_), d_out(d_out_), n(n_), n_heads(n_heads_) {
        pos1d       = register_parameter("pos1d",    torch::randn({n, d_tok}) * 0.02);
        queries     = register_parameter("queries",  torch::randn({K, d_tok}) * 0.02);
        query_pe    = register_parameter("query_pe", torch::randn({K, d_tok}) * 0.02);
        axial_block = register_module("axial_block", PreLNAttnBlock(d_tok, n_heads));

        type_embed      = register_parameter("type_embed",      torch::randn({2, d_tok}) * 0.02);
        token_idx_embed = register_parameter("token_idx_embed", torch::randn({K, d_tok}) * 0.02);
        row_coord_embed = register_parameter("row_coord_embed", torch::randn({n, d_tok}) * 0.02);
        col_coord_embed = register_parameter("col_coord_embed", torch::randn({n, d_tok}) * 0.02);
        fusion_block    = register_module("fusion_block", PreLNAttnBlock(d_tok, n_heads));

        out_proj = register_module("out_proj", torch::nn::Linear(d_tok, d_out));
        if (in_channels != d_tok)
            proj = register_module("proj", torch::nn::Linear(in_channels, d_tok));
    }

    torch::Tensor axial_summarize(torch::Tensor seq) {
        auto B = seq.size(0), n_lines = seq.size(1), L = seq.size(2);
        auto flat = seq.reshape({B * n_lines, L, d_tok});
        auto q = (queries + query_pe).unsqueeze(0).expand({B * n_lines, K, d_tok});
        auto tok = axial_block->forward_cross(q, flat);
        return tok.reshape({B, n_lines, K, d_tok});
    }

    torch::Tensor masked_mean_pool(torch::Tensor x, torch::Tensor mask) {
        auto mask_ = mask.to(x.dtype()).view({1, -1, 1});
        auto summed = (x * mask_).sum(1);
        auto denom = mask_.sum(1).clamp_min(1e-6);
        return summed / denom;
    }

    torch::Tensor fuse_selected(torch::Tensor row_tok, torch::Tensor col_tok,
                                torch::Tensor row_sel_idx, torch::Tensor col_sel_idx,
                                torch::Tensor row_coord_ids, torch::Tensor col_coord_ids) {
        auto B = row_tok.size(0);
        auto M = row_sel_idx.size(0);

        auto row_sel = row_tok.index_select(1, row_sel_idx);
        auto col_sel = col_tok.index_select(1, col_sel_idx);

        auto row_coord = row_coord_embed.index_select(0, row_coord_ids);
        auto col_coord = col_coord_embed.index_select(0, col_coord_ids);

        auto row_label = type_embed[0].view({1,1,d_tok}) + token_idx_embed.view({1,K,d_tok})
                        + row_coord.view({M,1,d_tok});
        auto col_label = type_embed[1].view({1,1,d_tok}) + token_idx_embed.view({1,K,d_tok})
                        + col_coord.view({M,1,d_tok});

        row_sel = row_sel + row_label.unsqueeze(0);
        col_sel = col_sel + col_label.unsqueeze(0);

        auto fusion_seq = torch::cat({row_sel, col_sel}, 2);
        fusion_seq = fusion_seq.reshape({B * M, 2 * K, d_tok});

        fusion_seq = fusion_block->forward_self(fusion_seq);

        auto mask = torch::ones({2 * K}, fusion_seq.options());
        auto pooled = masked_mean_pool(fusion_seq, mask);

        auto z = out_proj->forward(pooled);
        return z.reshape({B, M, d_out});
    }

    struct SparseGrid {
        torch::Tensor values;
        torch::Tensor cell_index;
        std::vector<int64_t> unique_rows, unique_cols;
    };

    SparseGrid forward_stage1(torch::Tensor x,
                          const std::vector<std::pair<int64_t,int64_t>>& coords) {
        if (proj) x = proj->forward(x.permute({0,2,3,1})).permute({0,3,1,2});

        auto rows = x.permute({0,2,3,1}) + pos1d.view({1,1,n,d_tok});
        auto cols = x.permute({0,3,2,1}) + pos1d.view({1,1,n,d_tok});

        auto combined_seq = torch::cat({rows, cols}, 1);
        auto combined = axial_summarize(combined_seq);

        auto row_tok = combined.narrow(1, 0, n);
        auto col_tok = combined.narrow(1, n, n);

        std::set<int64_t> rset, cset;
        for (auto& c : coords) { rset.insert(c.first); cset.insert(c.second); }
        std::vector<int64_t> unique_rows(rset.begin(), rset.end());
        std::vector<int64_t> unique_cols(cset.begin(), cset.end());

        std::set<std::pair<int64_t,int64_t>> cell_set;
        for (auto i : unique_rows)
            for (int64_t j = 0; j < n; ++j) cell_set.insert({i, j});
        for (auto j : unique_cols)
            for (int64_t i = 0; i < n; ++i) cell_set.insert({i, j});

        std::vector<int64_t> rows_idx, cols_idx;
        rows_idx.reserve(cell_set.size()); cols_idx.reserve(cell_set.size());
        for (auto& p : cell_set) { rows_idx.push_back(p.first); cols_idx.push_back(p.second); }

        auto ridx_t = torch::tensor(rows_idx, torch::kLong).to(x.device());
        auto cidx_t = torch::tensor(cols_idx, torch::kLong).to(x.device());
        auto values = fuse_selected(row_tok, col_tok, ridx_t, cidx_t, ridx_t, cidx_t);

        auto cell_index = torch::full({n, n}, -1, torch::kLong);
        for (size_t k = 0; k < rows_idx.size(); ++k)
            cell_index[rows_idx[k]][cols_idx[k]] = (int64_t)k;

        return {values, cell_index, unique_rows, unique_cols};
    }

    torch::Tensor forward_stage2(const SparseGrid& g,
                                 const std::vector<std::pair<int64_t,int64_t>>& coords) {
        auto B = g.values.size(0);
        torch::Tensor v = g.values;
        if (proj) v = proj->forward(v);

        auto row_ids_t = torch::tensor(g.unique_rows, torch::kLong);
        auto col_ids_t = torch::tensor(g.unique_cols, torch::kLong);

        auto row_gather = g.cell_index.index_select(0, row_ids_t);
        auto col_gather = g.cell_index.index_select(1, col_ids_t).transpose(0,1);

        row_gather = row_gather.to(v.device());
        col_gather = col_gather.to(v.device());

        auto row_seq = v.index({torch::indexing::Slice(), row_gather});
        auto col_seq = v.index({torch::indexing::Slice(), col_gather});

        row_seq = row_seq + pos1d.view({1,1,n,d_tok});
        col_seq = col_seq + pos1d.view({1,1,n,d_tok});

        auto combined_seq = torch::cat({row_seq, col_seq}, 1);
        auto combined = axial_summarize(combined_seq);

        auto row_tok = combined.narrow(1, 0, row_seq.size(1));
        auto col_tok = combined.narrow(1, row_seq.size(1), col_seq.size(1));
    
        std::map<int64_t,int64_t> row_pos, col_pos;
        for (size_t k=0; k<g.unique_rows.size(); ++k) row_pos[g.unique_rows[k]] = (int64_t)k;
        for (size_t k=0; k<g.unique_cols.size(); ++k) col_pos[g.unique_cols[k]] = (int64_t)k;

        std::vector<int64_t> r_sel, c_sel, r_id, c_id;
        for (auto& pc : coords) {
            r_sel.push_back(row_pos.at(pc.first));
            c_sel.push_back(col_pos.at(pc.second));
            r_id.push_back(pc.first);
            c_id.push_back(pc.second);
        }

        auto r_sel_t = torch::tensor(r_sel, torch::kLong).to(v.device());
        auto c_sel_t = torch::tensor(c_sel, torch::kLong).to(v.device());
        auto r_id_t  = torch::tensor(r_id,  torch::kLong).to(v.device());
        auto c_id_t  = torch::tensor(c_id,  torch::kLong).to(v.device());

        return fuse_selected(row_tok, col_tok, r_sel_t, c_sel_t, r_id_t, c_id_t);
    }
};
TORCH_MODULE(AFCStageSparse);

struct AFCBackboneSparseImpl : torch::nn::Module {
    AFCStageSparse stage1{nullptr}, stage2{nullptr};

    AFCBackboneSparseImpl(int64_t C = 32, int64_t d_tok1 = 128, int64_t d_out1 = 128,
                          int64_t d_tok2 = 256, int64_t d_out2 = 256, int64_t n = 31,
                          int64_t K = 6) {
        stage1 = register_module("stage1", AFCStageSparse(C, d_tok1, d_out1, n, K, 4));
        stage2 = register_module("stage2", AFCStageSparse(d_out1, d_tok2, d_out2, n, K, 8));
    }

    torch::Tensor forward(torch::Tensor x,
        const std::vector<std::pair<int64_t,int64_t>>& coords
            = {std::pair<int64_t,int64_t>{15, 15}}) {
        auto g1 = stage1->forward_stage1(x, coords);
        return stage2->forward_stage2(g1, coords);
    }
};
TORCH_MODULE(AFCBackboneSparse);

struct PlayerPolicyNetImpl : torch::nn::Module {
    static constexpr int64_t FIXED_ROW   = 15, FIXED_COL = 15;
    static constexpr int64_t WINDOW_SIZE = 31, PRED_HEADS = 31;
    static constexpr int64_t d_model     = 300;

    int64_t n, C;
    AFCBackboneSparse afc{nullptr};
    torch::nn::Linear  proj_in{nullptr};
    torch::nn::LayerNorm proj_norm{nullptr};
    PreLNAttnBlock tf_policy{nullptr};
    torch::Tensor cls_token, pos_embed;
    torch::nn::Linear policy_head{nullptr};
    
    std::vector<torch::nn::Linear> pred_heads;

    std::deque<torch::Tensor> buffer_queue;
    int64_t buffer_batch_size = -1;

    PlayerPolicyNetImpl(int64_t C_ = 32, int64_t n_ = 31, int64_t afc_d_out2 = 256)
        : n(n_), C(C_) {
        afc = register_module("afc", AFCBackboneSparse(C, 128, 128, 256, afc_d_out2, n, 6));

        for (int i = 0; i < PRED_HEADS; ++i)
            pred_heads.push_back(
                register_module("pred_head_" + std::to_string(i), torch::nn::Linear(afc_d_out2, N_ACTIONS))
            );

        int64_t concat_dim = afc_d_out2 + C + N_ACTIONS;
        proj_in   = register_module("proj_in",  torch::nn::Linear(concat_dim, d_model));
        proj_norm = register_module("proj_norm", torch::nn::LayerNorm(
                        torch::nn::LayerNormOptions({d_model})));

        tf_policy = register_module("tf_policy", PreLNAttnBlock(d_model, 10));

        cls_token = register_parameter("cls_token", torch::randn({1, 1, d_model}) * 0.02);
        pos_embed = register_parameter("pos_embed",
                        torch::randn({1, 1 + WINDOW_SIZE, d_model}) * 0.02);

        policy_head = register_module("policy_head", torch::nn::Linear(d_model, N_ACTIONS));

        reset_memory();
    }

    void reset_memory() {
        buffer_queue.clear();
        buffer_batch_size = -1;
    }

    std::vector<torch::Tensor> forward(
        torch::Tensor x, torch::Tensor prev_action, torch::Tensor active = {}) {

        namespace I = torch::indexing;
        auto B = x.size(0);
        auto device = x.device();

        if (buffer_batch_size != B) {
            reset_memory();
            buffer_batch_size = B;
        }

        auto afc_out = afc->forward(x);
        auto A_vec   = afc_out.squeeze(1);

        std::vector<torch::Tensor> prog;
        for (auto &pred_head: pred_heads)
            prog.push_back(pred_head->forward(A_vec).unsqueeze(1));

        auto pred = torch::stack(prog, 1).squeeze(2);

        auto B_vec   = x.index({I::Slice(), I::Slice(),
                                FIXED_ROW, FIXED_COL}).clone();
        auto C_vec   = torch::one_hot(prev_action, N_ACTIONS)
                           .to(A_vec.dtype());
        auto x_cat   = torch::cat({A_vec.detach(), B_vec, C_vec}, 1);

        if (active.defined()) {
            x_cat = x_cat * active.to(x_cat.dtype()).unsqueeze(-1);
        }

        buffer_queue.push_back(x_cat);
        if ((int64_t)buffer_queue.size() > WINDOW_SIZE)
            buffer_queue.pop_front();

        std::vector<torch::Tensor> raw_frames(buffer_queue.begin(), buffer_queue.end()), frames;

        for (int i = 0; i < raw_frames.size(); ++i)
            frames.push_back(proj_norm->forward(proj_in->forward(raw_frames[i])));

        auto seq_buffer = torch::stack(frames, 1);
        int64_t L_hist = seq_buffer.size(1);

        auto cls = cls_token.expand({B, 1, d_model});
        auto seq = torch::cat({cls, seq_buffer}, 1);
        int64_t L = L_hist + 1;
        seq = seq + pos_embed.index({I::Slice(), I::Slice(0, L)});

        auto out     = tf_policy->forward_self(seq);
        auto cls_out = out.index({I::Slice(), 0});
        auto logits  = policy_head->forward(cls_out);

        return {pred, logits};
    }
};
TORCH_MODULE(PlayerPolicyNet);

// -----------------------------------------------------------------------
//  Focal loss
// -----------------------------------------------------------------------
torch::Tensor focal_loss(torch::Tensor logits, torch::Tensor targets, double gamma) {
    auto log_probs = torch::log_softmax(logits, 1);
    auto target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1);
    auto p_t = torch::exp(target_log_probs);
    auto focal_weight = torch::pow(1 - p_t, gamma);
    auto loss = -focal_weight * target_log_probs;
    return loss;
}

torch::Tensor future_pred_loss(
    torch::Tensor pred,
    torch::Tensor future_actions,
    double gamma_future = 0.9,
    double gamma = 2.0
) {
    auto H = pred.size(1);

    auto weights = torch::pow(gamma_future, torch::arange(H, pred.options()))
                        .view({1, H, 1});

    auto log_probs = torch::log_softmax(pred, -1);

    auto gathered = log_probs.gather(-1, future_actions.unsqueeze(-1)).squeeze(-1);
    auto loss_per_step = -gathered * weights.squeeze(-1);

    auto loss = loss_per_step.sum();
    return loss;
}

torch::Tensor future_pred_loss_per_sample(
    torch::Tensor pred,
    torch::Tensor future_actions,
    double gamma_future = 0.9,
    double gamma = 2.0
) {
    auto H = pred.size(1);

    auto weights = torch::pow(gamma_future, torch::arange(H, pred.options()))
                        .view({1, H, 1});

    auto log_probs = torch::log_softmax(pred, -1);

    auto gathered = log_probs.gather(-1, future_actions.unsqueeze(-1)).squeeze(-1);
    auto loss_per_step = -gathered * weights.squeeze(-1);

    return loss_per_step.sum(1);
}

// -----------------------------------------------------------------------
//  Checkpointing helpers
// -----------------------------------------------------------------------
void save_checkpoint(PlayerPolicyNet& model,
                     std::unique_ptr<torch::optim::AdamW>& optimizer,
                     int epoch,
                     double best_val_loss,
                     const std::string& model_path,
                     const std::string& optim_path,
                     const std::string& meta_path) {
    model->reset_memory();
    torch::save(model, model_path);
    torch::save(*optimizer, optim_path);

    std::ofstream meta(meta_path);
    meta << epoch << "\n" << best_val_loss << "\n";
}

std::tuple<int, double> load_checkpoint(
        PlayerPolicyNet& model,
        torch::Device device,
        const std::string& model_path,
        const std::string& optim_path,
        const std::string& meta_path,
        std::unique_ptr<torch::optim::AdamW>& optimizer_out) {

    torch::serialize::InputArchive model_archive;
    model_archive.load_from(model_path, device);
    model->load(model_archive);
    model->to(device);

    optimizer_out = std::make_unique<torch::optim::AdamW>(
        model->parameters(),
        torch::optim::AdamWOptions(1e-3).weight_decay(1e-4));

    torch::serialize::InputArchive optim_archive;
    optim_archive.load_from(optim_path, device);
    optimizer_out->load(optim_archive);

    int epoch;
    double best_val_loss;
    std::ifstream meta(meta_path);
    meta >> epoch >> best_val_loss;
    return {epoch, best_val_loss};
}

// ---------- helper: load a single episode from disk ----------
std::pair<torch::Tensor, torch::Tensor> load_episode(const std::string& file_path,
                                                     torch::Device device) {
    torch::serialize::InputArchive archive;
    archive.load_from(file_path);
    
    torch::Tensor states, actions;
    
    archive.read("states", states);
    archive.read("actions", actions);

    return {states.to(device),
            actions.to(device)};
}

// ---------- scale gradients by valid count ----------
void scale_gradients(PlayerPolicyNet& model, int64_t n) {
    if (n == 0) return;
    auto params = model->parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        auto& param = params[i];
        if (param.grad().defined()) {
            param.mutable_grad() = param.grad() / n;
        }
    }
}

// ---------- process a batch of episodes in parallel ----------
void process_batch_gpu(PlayerPolicyNet& model,
                       const std::vector<std::string>& batch_files,
                       torch::Device device,
                       double gamma,
                       int64_t padding,
                       double gamma_future,
                       double& out_loss_sum,
                       int64_t& out_valid_count) {
    
    int64_t B = batch_files.size();
    if (B == 0) {
        out_loss_sum = 0.0;
        out_valid_count = 0;
        return;
    }

    // 1. Load all episodes in the batch
    std::vector<torch::Tensor> states_list, actions_list;
    std::vector<int64_t> lengths_cpu;
    int64_t T_max = 0;
    int64_t C = 0, H = 0, W = 0;

    for (const auto& f : batch_files) {
        auto [states, actions] = load_episode(f, device);
        states = states.squeeze(0);  // [T, C, H, W]
        actions = actions.squeeze(0); // [T]
        int64_t T_i = states.size(0);
        lengths_cpu.push_back(T_i);
        T_max = std::max(T_max, T_i);
        if (C == 0) {
            C = states.size(1);
            H = states.size(2);
            W = states.size(3);
        }
        states_list.push_back(states);
        actions_list.push_back(actions);
    }

    // 2. Pad to [B, T_max, C, H, W] and [B, T_max]
    auto states_padded = torch::zeros({B, T_max, C, H, W}, torch::dtype(torch::kFloat32).device(device));
    auto actions_padded = torch::zeros({B, T_max}, torch::dtype(torch::kLong).device(device));

    for (int64_t i = 0; i < B; ++i) {
        int64_t T_i = lengths_cpu[i];
        states_padded.index_put_({i, torch::indexing::Slice(0, T_i)}, states_list[i]);
        actions_padded.index_put_({i, torch::indexing::Slice(0, T_i)}, actions_list[i]);
    }

    auto lengths = torch::tensor(lengths_cpu, torch::dtype(torch::kLong).device(device));

    // 3. Precompute future loss normalisation
    double sum_w = 0.0, tmp = 1.0;
    for (int i = 0; i < padding + 1; ++i) {
        sum_w += tmp;
        tmp *= gamma_future;
    }

    model->reset_memory();
    auto prev_action = torch::full({B}, 5, torch::dtype(torch::kLong).device(device));

    double batch_loss_sum = 0.0;
    int64_t total_valid_count = 0;

    // 4. Loop over timesteps
    for (int64_t t = 0; t < T_max; ++t) {
        auto x = states_padded.select(1, t);  // [B, C, H, W]
        auto active = (lengths > t);           // [B] boolean

        auto out = model->forward(x, prev_action, active);
        auto pred = out[0];    // [B, PRED_HEADS, N_ACTIONS]
        auto logits = out[1];  // [B, N_ACTIONS]

        torch::Tensor step_loss = torch::zeros({1}, device);

        // Policy loss (t >= padding)
        if (t >= padding) {
            auto target = actions_padded.select(1, t);  // [B]
            auto pol_loss = focal_loss(logits, target, gamma);  // [B]
            auto valid = (lengths > t).to(pol_loss.dtype());    // [B]
            step_loss = step_loss + (pol_loss * valid).sum();
            total_valid_count += valid.sum().item<int64_t>();
        }

        // Future loss (t < T_max - padding)
        if (t < T_max - padding) {
            auto future_targets = actions_padded.slice(1, t, t + padding); // [B, padding]
            auto aux_loss = future_pred_loss_per_sample(pred, future_targets, gamma_future, gamma); // [B]
            auto future_valid = ((lengths - padding) > t).to(aux_loss.dtype()); // [B]
            step_loss = step_loss + (aux_loss * future_valid).sum() / sum_w;
        }

        // Backward for this timestep (gradient accumulation)
        if (step_loss.defined() && step_loss.numel() > 0 && step_loss.item<double>() != 0.0) {
            step_loss.backward();
            batch_loss_sum += step_loss.item<double>();
        }

        prev_action = actions_padded.select(1, t);  // [B]
    }

    model->reset_memory();

    out_loss_sum = batch_loss_sum;
    out_valid_count = total_valid_count;
}

struct PolicyMetrics {
    std::vector<int64_t> topk_correct;
    std::vector<std::vector<int64_t>> conf_mat;
    PolicyMetrics() : topk_correct(N_ACTIONS - 1, 0),
                      conf_mat(N_ACTIONS,
                               std::vector<int64_t>(N_ACTIONS, 0)) {}
};

struct EvalResult {
    double loss_sum;
    int64_t valid_count;
    PolicyMetrics main_policy;
    std::vector<PolicyMetrics> policy_metrics;
    std::vector<int64_t> future_correct;
    std::vector<int64_t> future_total;
};

// -----------------------------------------------------------------------
//  Evaluate a batch of episodes in parallel (no gradients)
// -----------------------------------------------------------------------
EvalResult evaluate_batch(PlayerPolicyNet& model,
                          const std::vector<std::string>& batch_files,
                          torch::Device device,
                          double gamma,
                          int64_t padding,
                          double gamma_future = 0.9) {
    int64_t B = batch_files.size();
    if (B == 0) {
        EvalResult empty;
        empty.valid_count = 0;
        empty.loss_sum = 0.0;
        empty.main_policy = PolicyMetrics();
        empty.policy_metrics.assign(padding, PolicyMetrics());
        empty.future_correct.assign(padding, 0);
        empty.future_total.assign(padding, 0);
        return empty;
    }

    // Load all episodes in the batch
    std::vector<torch::Tensor> states_list, actions_list;
    std::vector<int64_t> lengths_cpu;
    int64_t T_max = 0;
    int64_t C = 0, H = 0, W = 0;

    for (const auto& f : batch_files) {
        auto [states, actions] = load_episode(f, device);
        states = states.squeeze(0);  // [T, C, H, W]
        actions = actions.squeeze(0); // [T]
        int64_t T_i = states.size(0);
        lengths_cpu.push_back(T_i);
        T_max = std::max(T_max, T_i);
        if (C == 0) {
            C = states.size(1);
            H = states.size(2);
            W = states.size(3);
        }
        states_list.push_back(states);
        actions_list.push_back(actions);
    }

    // Pad to [B, T_max, C, H, W] and [B, T_max]
    auto states_padded = torch::zeros({B, T_max, C, H, W}, torch::dtype(torch::kFloat32).device(device));
    auto actions_padded = torch::zeros({B, T_max}, torch::dtype(torch::kLong).device(device));

    for (int64_t i = 0; i < B; ++i) {
        int64_t T_i = lengths_cpu[i];
        states_padded.index_put_({i, torch::indexing::Slice(0, T_i)}, states_list[i]);
        actions_padded.index_put_({i, torch::indexing::Slice(0, T_i)}, actions_list[i]);
    }

    auto lengths = torch::tensor(lengths_cpu, torch::dtype(torch::kLong).device(device));

    // Precompute future loss normalisation constant
    double sum_w = 0.0, tmp = 1.0;
    for (int i = 0; i < padding + 1; ++i) {
        sum_w += tmp;
        tmp *= gamma_future;
    }

    model->reset_memory();
    auto prev_action = torch::full({B}, 5, torch::dtype(torch::kLong).device(device));
    torch::NoGradGuard no_grad;

    EvalResult r;
    r.loss_sum = 0.0;
    r.valid_count = 0;
    r.main_policy = PolicyMetrics();
    r.policy_metrics.assign(padding, PolicyMetrics());
    r.future_correct.assign(padding, 0);
    r.future_total.assign(padding, 0);

    // Loop over timesteps
    for (int64_t t = 0; t < T_max; ++t) {
        auto x = states_padded.select(1, t);  // [B, C, H, W]
        auto active = (lengths > t);           // [B] boolean

        auto out = model->forward(x, prev_action, active);
        auto pred = out[0];    // [B, PRED_HEADS, N_ACTIONS]
        auto logits = out[1];  // [B, N_ACTIONS]

        torch::Tensor step_loss = torch::zeros({1}, device);

        // ---- Main policy ----
        if (t >= padding) {
            auto target = actions_padded.select(1, t);  // [B]
            auto pol_loss = focal_loss(logits, target, gamma);  // [B]
            auto valid = (lengths > t).to(pol_loss.dtype());    // [B]
            step_loss = step_loss + (pol_loss * valid).sum();

            // Collect metrics
            auto argmax = logits.argmax(1);        // [B]
            auto topk_res = torch::topk(logits, N_ACTIONS - 1, 1);
            auto topk_idx = std::get<1>(topk_res); // [B, N_ACTIONS-1]

            for (int64_t i = 0; i < B; ++i) {
                if (valid[i].item<bool>()) {
                    int64_t gt = target[i].item<int64_t>();
                    int64_t pred_val = argmax[i].item<int64_t>();
                    r.main_policy.conf_mat[gt][pred_val]++;

                    for (int k = 1; k <= N_ACTIONS - 1; ++k) {
                        bool correct = false;
                        for (int j = 0; j < k; ++j) {
                            if (topk_idx[i][j].item<int64_t>() == gt) {
                                correct = true;
                                break;
                            }
                        }
                        if (correct) r.main_policy.topk_correct[k-1]++;
                    }
                }
            }
            r.valid_count += valid.sum().item<int64_t>();
        }

        // ---- Future heads ----
        if (t < T_max - padding) {
            auto future_targets = actions_padded.slice(1, t, t + padding); // [B, padding]
            auto aux_loss = future_pred_loss(pred, future_targets, gamma_future); // scalar (sum over batch)
            step_loss += aux_loss / sum_w;

            auto pred_argmax = pred.argmax(-1); // [B, padding]
            auto future_valid = ((lengths - padding) > t).to(torch::kLong);

            for (int64_t i = 0; i < B; ++i) {
                if (future_valid[i].item<bool>()) {
                    for (int64_t h = 0; h < padding; ++h) {
                        int64_t gt = future_targets[i][h].item<int64_t>();
                        int64_t pred_h = pred_argmax[i][h].item<int64_t>();
                        r.future_total[h] += 1;
                        if (pred_h == gt) r.future_correct[h] += 1;

                        auto logits_h = pred[i][h]; // [N_ACTIONS]
                        auto topk_res = torch::topk(logits_h, N_ACTIONS - 1, 0);
                        auto topk_idx = std::get<1>(topk_res);
                        for (int k = 1; k <= N_ACTIONS - 1; ++k) {
                            bool correct = false;
                            for (int j = 0; j < k; ++j) {
                                if (topk_idx[j].item<int64_t>() == gt) {
                                    correct = true;
                                    break;
                                }
                            }
                            if (correct) r.policy_metrics[h].topk_correct[k-1]++;
                        }
                        r.policy_metrics[h].conf_mat[gt][pred_h]++;
                    }
                }
            }
        }

        if (step_loss.defined() && step_loss.numel() > 0)
            r.loss_sum += step_loss.item<double>();

        prev_action = actions_padded.select(1, t);
    }

    model->reset_memory();
    return r;
}

// -----------------------------------------------------------------------
//  Helper to print per‑class metrics
// -----------------------------------------------------------------------
void print_class_metrics(const std::vector<std::vector<int64_t>>& conf_mat,
                         int64_t total_samples,
                         const std::string& prefix = "") {
    int num_classes = conf_mat.size();
    std::cout << prefix << "Per-class metrics:\n";
    std::cout << prefix << "Class\tPrecision\tRecall\t\tF1-score\tSupport\n";
    for (int i = 0; i < num_classes; ++i) {
        int64_t TP = conf_mat[i][i];
        int64_t FP = 0, FN = 0;
        for (int j = 0; j < num_classes; ++j) {
            if (j != i) FP += conf_mat[j][i];
            if (j != i) FN += conf_mat[i][j];
        }
        int64_t support = TP + FN;
        double precision = (TP + FP > 0) ? (double)TP / (TP + FP) : 0.0;
        double recall    = (support > 0) ? (double)TP / support : 0.0;
        double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;

        std::cout << prefix << std::setw(5) << i << "\t"
                  << std::fixed << std::setprecision(4) << precision << "\t\t"
                  << recall << "\t\t" << f1 << "\t\t"
                  << support << "\n";
    }
    int64_t correct_total = 0;
    for (int i = 0; i < num_classes; ++i) correct_total += conf_mat[i][i];
    double overall_acc = (total_samples > 0) ? (double)correct_total / total_samples : 0.0;
    std::cout << prefix << "Overall accuracy: " << std::fixed << std::setprecision(4) << overall_acc << "\n";
}

// -----------------------------------------------------------------------
//  Validation using batches (parallel on GPU)
// -----------------------------------------------------------------------
double validation(PlayerPolicyNet& model,
                  std::unique_ptr<torch::optim::AdamW>& optimizer,
                  const std::vector<std::string>& val_files,
                  int epoch,
                  double& best_val_loss,
                  const std::string& model_path,
                  const std::string& optim_path,
                  const std::string& meta_path,
                  const torch::Device& device,
                  double gamma,
                  int64_t padding,
                  double gamma_future = 0.9) {

    const int64_t VAL_BATCH_SIZE = 16;

    double val_loss_sum = 0.0;
    int64_t val_valid_cnt = 0;

    PolicyMetrics total_main_policy;
    std::vector<PolicyMetrics> total_future_policy(padding);
    std::vector<int64_t> total_future_correct(padding, 0);
    std::vector<int64_t> total_future_total(padding, 0);

    // Process validation files in batches
    for (size_t start = 0; start < val_files.size(); start += VAL_BATCH_SIZE) {
        size_t end = std::min(start + VAL_BATCH_SIZE, val_files.size());
        std::vector<std::string> batch_files(val_files.begin() + start, val_files.begin() + end);

        auto result = evaluate_batch(model, batch_files, device, gamma, padding, gamma_future);

        val_loss_sum += result.loss_sum;
        val_valid_cnt += result.valid_count;

        // Accumulate main policy stats
        for (size_t k = 0; k < total_main_policy.topk_correct.size(); ++k)
            total_main_policy.topk_correct[k] += result.main_policy.topk_correct[k];
        for (int i = 0; i < N_ACTIONS; ++i)
            for (int j = 0; j < N_ACTIONS; ++j)
                total_main_policy.conf_mat[i][j] += result.main_policy.conf_mat[i][j];

        // Accumulate future heads stats
        for (int64_t h = 0; h < padding; ++h) {
            for (size_t k = 0; k < total_future_policy[h].topk_correct.size(); ++k)
                total_future_policy[h].topk_correct[k] += result.policy_metrics[h].topk_correct[k];
            for (int i = 0; i < N_ACTIONS; ++i)
                for (int j = 0; j < N_ACTIONS; ++j)
                    total_future_policy[h].conf_mat[i][j] += result.policy_metrics[h].conf_mat[i][j];
            total_future_correct[h] += result.future_correct[h];
            total_future_total[h]   += result.future_total[h];
        }
    }

    double val_avg = 0.0;
    if (val_valid_cnt > 0) {
        val_avg = val_loss_sum / val_valid_cnt;

        // Print results
        double main_acc1 = (double)total_main_policy.topk_correct[0] / val_valid_cnt;
        std::cout << "\n=== Epoch " << epoch
                  << " val loss: " << val_avg
                  << " | main policy top-1 acc: " << main_acc1
                  << " (" << val_valid_cnt << " decisions) ===" << std::endl;

        std::cout << "  Main policy top-k accuracies:" << std::endl;
        for (int k = 1; k <= N_ACTIONS - 1; ++k) {
            double acc_k = (double)total_main_policy.topk_correct[k-1] / val_valid_cnt;
            std::cout << "    top-" << k << ": " << acc_k << std::endl;
        }

        std::vector<double> f1_per_class(N_ACTIONS, 0.0);
        for (int i = 0; i < N_ACTIONS; ++i) {
            int64_t tp = total_main_policy.conf_mat[i][i];
            int64_t fp = 0, fn = 0;
            for (int j = 0; j < N_ACTIONS; ++j) {
                if (j != i) fp += total_main_policy.conf_mat[j][i];
                if (j != i) fn += total_main_policy.conf_mat[i][j];
            }
            double precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
            double recall    = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
            double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
            f1_per_class[i] = f1;
        }
        double macro_f1 = std::accumulate(f1_per_class.begin(), f1_per_class.end(), 0.0) / N_ACTIONS;
        std::cout << "  Main policy Macro F1-score: " << macro_f1 << std::endl;

        print_class_metrics(total_main_policy.conf_mat, val_valid_cnt, "  ");

        for (int64_t h = 0; h < padding; ++h) {
            if (total_future_total[h] == 0) continue;
            std::cout << "\n  Future head " << h << " (horizon " << (h+1) << "):" << std::endl;
            double acc1 = (double)total_future_correct[h] / total_future_total[h];
            std::cout << "    top-1 accuracy: " << acc1 << std::endl;
            std::cout << "    Top-k accuracies:" << std::endl;
            for (int k = 1; k <= N_ACTIONS - 1; ++k) {
                double acc_k = (double)total_future_policy[h].topk_correct[k-1] / total_future_total[h];
                std::cout << "      top-" << k << ": " << acc_k << std::endl;
            }
            std::vector<double> f1_per_class_h(N_ACTIONS, 0.0);
            for (int i = 0; i < N_ACTIONS; ++i) {
                int64_t tp = total_future_policy[h].conf_mat[i][i];
                int64_t fp = 0, fn = 0;
                for (int j = 0; j < N_ACTIONS; ++j) {
                    if (j != i) fp += total_future_policy[h].conf_mat[j][i];
                    if (j != i) fn += total_future_policy[h].conf_mat[i][j];
                }
                double precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
                double recall    = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
                double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
                f1_per_class_h[i] = f1;
            }
            double macro_f1_h = std::accumulate(f1_per_class_h.begin(), f1_per_class_h.end(), 0.0) / N_ACTIONS;
            std::cout << "    Macro F1-score: " << macro_f1_h << std::endl;

            print_class_metrics(total_future_policy[h].conf_mat, total_future_total[h], "    ");
        }

        if (val_avg < best_val_loss) {
            best_val_loss = val_avg;
            save_checkpoint(model, optimizer, epoch, best_val_loss,
                            model_path, optim_path, meta_path);
            std::cout << "Checkpoint saved (new best validation loss)." << std::endl;
        }
    } else {
        std::cout << "Validation: no valid decisions." << std::endl;
    }

    return val_avg;
}

// -----------------------------------------------------------------------
//  Main training
// -----------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string data_dir = "../dataset/data_train";
    std::string val_dir = "../dataset/data_val";
    int num_epochs = 512;

    if (argc > 1) data_dir = argv[1];
    if (argc > 2) num_epochs = std::stoi(argv[2]);
    if (argc > 3) val_dir = argv[3];

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << device << std::endl;

    std::vector<std::string> episode_files;
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.path().extension() == ".pt")
            episode_files.push_back(entry.path().string());
    }
    if (episode_files.empty()) {
        std::cerr << "No .pt files found in " << data_dir << std::endl;
        return 1;
    }
    std::cout << "Found " << episode_files.size() << " episodes." << std::endl;

    std::vector<std::string> val_files;
    if (fs::exists(val_dir)) {
        for (const auto& entry : fs::directory_iterator(val_dir)) {
            if (entry.path().extension() == ".pt")
                val_files.push_back(entry.path().string());
        }
        std::cout << "Found " << val_files.size() << " validation episodes." << std::endl;
    } else {
        std::cout << "Validation directory not found, skipping validation." << std::endl;
    }

    PlayerPolicyNet model{nullptr};
    std::unique_ptr<torch::optim::AdamW> optimizer{nullptr};

    model = PlayerPolicyNet();

    const std::string model_path = "../backup/model.pt";
    const std::string optim_path = "../backup/optimizer.pt";
    const std::string meta_path  = "../backup/meta.txt";

    int start_epoch = 0;
    double best_val_loss = std::numeric_limits<double>::infinity();

    if (fs::exists(model_path) && fs::exists(optim_path) && fs::exists(meta_path)) {
        std::tie(start_epoch, best_val_loss) =
            load_checkpoint(model, device, model_path, optim_path, meta_path, optimizer);
        start_epoch += 1;
        std::cout << "Resumed from epoch " << (start_epoch-1)
                  << " with best loss " << best_val_loss << std::endl;
    } else {
        model->to(device);
        optimizer = std::make_unique<torch::optim::AdamW>(
            model->parameters(),
            torch::optim::AdamWOptions(1e-3).weight_decay(1e-4));
    }

    const int64_t BATCH_SIZE = 16;
    const int64_t PADDING = model->PRED_HEADS;
    const double GAMMA = 2.0;
    const double GAMMA_FUTURE = 0.9;

    std::vector<int64_t> indices(episode_files.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(std::random_device{}());

/*
    validation(model, optimizer, val_files, start_epoch, best_val_loss,
               model_path, optim_path, meta_path, device,
               GAMMA, PADDING, GAMMA_FUTURE);
    exit(0);
*/

    for (int epoch = start_epoch; epoch < num_epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), gen);

        double epoch_loss_sum = 0.0;
        int64_t epoch_valid_cnt = 0;
        int batches_done = 0;

        for (size_t start = 0; start + BATCH_SIZE <= indices.size(); start += BATCH_SIZE) {
            std::vector<std::string> batch_files;
            for (int64_t b = 0; b < BATCH_SIZE; ++b) {
                batch_files.push_back(episode_files[indices[start + b]]);
            }

            model->zero_grad();

            double batch_loss_sum = 0.0;
            int64_t batch_valid_total = 0;

            process_batch_gpu(model, batch_files, device, GAMMA, PADDING, GAMMA_FUTURE,
                              batch_loss_sum, batch_valid_total);

            if (batch_valid_total > 0) {
                scale_gradients(model, batch_valid_total);
                torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
                optimizer->step();
                model->zero_grad();

                double avg_loss = batch_loss_sum / batch_valid_total;
                epoch_loss_sum += batch_loss_sum;
                epoch_valid_cnt += batch_valid_total;
                batches_done++;

                std::cout << "Epoch " << epoch << " batch " << batches_done
                          << " avg loss: " << avg_loss << std::endl;
            } else {
                std::cerr << "Warning: batch had no valid actions, skipping." << std::endl;
                model->zero_grad();
            }
        }

        double val_avg = 0.0;
        if (!val_files.empty()) {
            val_avg = validation(model, optimizer, val_files, epoch, best_val_loss,
                                 model_path, optim_path, meta_path, device,
                                 GAMMA, PADDING, GAMMA_FUTURE);
        }

        if (epoch_valid_cnt > 0) {
            double epoch_avg = epoch_loss_sum / epoch_valid_cnt;
            std::cout << "=== Epoch " << epoch
                      << " training avg loss: " << epoch_avg
                      << " (" << epoch_valid_cnt << " decisions) ===" << std::endl;

            if (val_files.empty()) {
                best_val_loss = epoch_avg;
                save_checkpoint(model, optimizer, epoch, best_val_loss,
                                model_path, optim_path, meta_path);
                std::cout << "Checkpoint saved (new best training loss)." << std::endl;
            }
        } else {
            std::cout << "Epoch " << epoch << " had no valid training decisions." << std::endl;
        }
    }

    std::cout << "Training finished." << std::endl;
    return 0;
}
