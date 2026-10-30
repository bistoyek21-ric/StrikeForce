/*
MIT License
Copyright (c) 2025 bistoyek21 R.I.C.
... (same license header) ...
*/

#include "../../basic.hpp"

#pragma once

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
#include <deque>

namespace fs = std::filesystem;

static constexpr int64_t N_ACTIONS   = 7;

// =====================================================================
// PreLNAttnBlock (unchanged)
// =====================================================================
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

// =====================================================================
// AFCStageSparse / AFCBackboneSparse (unchanged)
// =====================================================================
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
        pos1d       = register_parameter("pos1d",       torch::randn({n, d_tok}) * 0.02);
        type_embed  = register_parameter("type_embed",  torch::randn({2, d_tok}) * 0.02);
        queries     = register_parameter("queries",  torch::randn({K, d_tok}) * 0.02);
        query_pe    = register_parameter("query_pe", torch::randn({K, d_tok}) * 0.02);
        axial_block = register_module("axial_block", PreLNAttnBlock(d_tok, n_heads));

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

        auto row_label = token_idx_embed.view({1,K,d_tok}) + row_coord.view({M,1,d_tok});
        auto col_label = token_idx_embed.view({1,K,d_tok}) + col_coord.view({M,1,d_tok});

        row_sel = row_sel + row_label.unsqueeze(0);
        col_sel = col_sel + col_label.unsqueeze(0);

        auto fusion_seq = torch::cat({row_sel, col_sel}, 2).reshape({B * M, 2 * K, d_tok});
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

        auto rows = x.permute({0,2,3,1}) + (pos1d + type_embed[0]).view({1,1,n,d_tok});
        auto cols = x.permute({0,3,2,1}) + (pos1d + type_embed[1]).view({1,1,n,d_tok});

        auto combined_seq = torch::cat({rows, cols}, 1);
        auto combined = axial_summarize(combined_seq);

        auto row_tok = combined.narrow(1, 0, n);
        auto col_tok = combined.narrow(1, n, n);

        std::set<int64_t> rset, cset;
        for (auto& c : coords) { rset.insert(c.first); cset.insert(c.second); }
        std::vector<int64_t> unique_rows(rset.begin(), rset.end());
        std::vector<int64_t> unique_cols(cset.begin(), cset.end());

        std::set<std::pair<int64_t,int64_t>> cell_set;
        for (auto i : unique_rows) for (int64_t j = 0; j < n; ++j) cell_set.insert({i, j});
        for (auto j : unique_cols) for (int64_t i = 0; i < n; ++i) cell_set.insert({i, j});

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

        row_seq = row_seq + (pos1d + type_embed[0]).view({1,1,n,d_tok});
        col_seq = col_seq + (pos1d + type_embed[1]).view({1,1,n,d_tok});

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

// =====================================================================
// PlayerPolicyNet (unchanged)
// =====================================================================
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
                register_module("pred_head_" + std::to_string(i),
                                torch::nn::Linear(afc_d_out2, N_ACTIONS)));

        int64_t concat_dim = afc_d_out2 + C + N_ACTIONS;
        proj_in   = register_module("proj_in",  torch::nn::Linear(concat_dim, d_model));
        proj_norm = register_module("proj_norm",
                        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        tf_policy = register_module("tf_policy", PreLNAttnBlock(d_model, 10));

        cls_token = register_parameter("cls_token",
                        torch::randn({1, 1, d_model}) * 0.02);
        pos_embed = register_parameter("pos_embed",
                        torch::randn({1, 1 + WINDOW_SIZE, d_model}) * 0.02);
        policy_head = register_module("policy_head",
                          torch::nn::Linear(d_model, N_ACTIONS));

        reset_memory();
    }

    void reset_memory() {
        buffer_queue.clear();
        buffer_batch_size = -1;
    }

private:
    std::vector<torch::Tensor> forward_impl(torch::Tensor x, torch::Tensor prev_action) {
        namespace I = torch::indexing;
        auto B = x.size(0);

        if (buffer_batch_size != B) { reset_memory(); buffer_batch_size = B; }

        auto afc_out = afc->forward(x);
        auto A_vec   = afc_out.squeeze(1);

        std::vector<torch::Tensor> prog;
        for (auto &pred_head : pred_heads)
            prog.push_back(pred_head->forward(A_vec).unsqueeze(1));
        auto pred = torch::stack(prog, 1).squeeze(2);

        auto B_vec = x.index({I::Slice(), I::Slice(), FIXED_ROW, FIXED_COL}).clone();
        auto C_vec = torch::one_hot(prev_action, N_ACTIONS).to(A_vec.dtype());
        auto x_cat = torch::cat({A_vec.detach(), B_vec, C_vec}, 1);

        buffer_queue.push_back(x_cat);
        if ((int64_t)buffer_queue.size() > WINDOW_SIZE) buffer_queue.pop_front();

        std::vector<torch::Tensor> raw_frames(buffer_queue.begin(), buffer_queue.end()), frames;
        for (size_t i = 0; i < raw_frames.size(); ++i)
            frames.push_back(proj_norm->forward(proj_in->forward(raw_frames[i])));

        auto seq_buffer = torch::stack(frames, 1);
        int64_t L_hist  = seq_buffer.size(1);

        auto cls = cls_token.expand({B, 1, d_model});
        auto seq = torch::cat({cls, seq_buffer}, 1);
        int64_t L = L_hist + 1;
        seq = seq + pos_embed.index({I::Slice(), I::Slice(0, L)});

        auto out     = tf_policy->forward_self(seq);
        auto cls_out = out.index({I::Slice(), 0});

        return {pred, cls_out};
    }

public:
    std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor prev_action) {
        auto r      = forward_impl(x, prev_action);
        auto logits = policy_head->forward(r[1]);
        return {r[0], logits};
    }

    std::vector<torch::Tensor> forward_r_t(torch::Tensor x, torch::Tensor prev_action) {
        return forward_impl(x, prev_action);
    }
};
TORCH_MODULE(PlayerPolicyNet);


// =====================================================================
// CommExtensionNet — inter-agent communication extension
// ---------------------------------------------------------------------
// Per-agent token layout:
//
//   [ r_t_i ,  tok_value ,  m_{j1→i} ,  m_{j2→i} , ... ,  m_{j_{n-1}→i} ]
//       ^         ^             ^  ^               ^
//       |         |             |  |               |
//       |         |             |  +---- (n-1) incoming messages from others
//       |         |             +-------- each gets sender_pe[j] on RAW,
//       |         |                        concat dist, then Linear → d_r
//       |         +---------------------- learnable, NO PE (learnable itself)
//       +-------------------------------- r_t with rt_pe (dedicated PE)
//
// Sequence length per agent: n + 1
// Total tokens: n agents × (n+1) tokens
//
// PE design (n+1 distinct slots):
//   * sender_pe  : [n, d_msg] — one per sender identity, on RAW messages
//   * rt_pe      : [d_r]      — DEDICATED to r_t, distinct from any sender PE
//   * tok_value  : learnable [d_r], no additional PE
//
// Output heads:
//   * r_t token output       → policy_head  → logits            [B, n, n_actions]
//   * tok_value token output → value_head   → scalar value       [B, n]
//   * message token outputs  → msg_out_head → m_{i→j}^t          [B, n, n, d_msg]
//
// Stateless. Buffers, two-pass algorithm, and message-injection loss
// are the caller's responsibility.
// =====================================================================
struct CommExtensionNetImpl : torch::nn::Module {
    int64_t n, d_r, d_msg, d_dist, n_actions;

    // PE tables & learnable tokens
    torch::Tensor sender_pe;   // [n, d_msg]  — per-sender identity PE (raw msgs)
    torch::Tensor rt_pe;       // [d_r]       — dedicated PE for r_t
    torch::Tensor tok_value;   // [d_r]       — learnable value token (no PE)

    // Projections & heads
    torch::nn::Linear msg_proj{nullptr};      // (d_msg + d_dist) → d_r
    torch::nn::Linear msg_out_head{nullptr};  // d_r → d_msg
    torch::nn::Linear value_head{nullptr};    // d_r → 1
    torch::nn::Linear policy_head{nullptr};   // d_r → n_actions

    // Transformer
    PreLNAttnBlock comm_tf{nullptr};

    // Constant gather index: other_idx[i] = [j != i]
    torch::Tensor other_idx;   // [n, n-1] Long, CPU

    struct Output {
        torch::Tensor new_messages;   // [B, n, n, d_msg]      (diagonal zeroed)
        torch::Tensor values;         // [B, n]
        torch::Tensor logits;         // [B, n, n_actions]
    };

    CommExtensionNetImpl(int64_t n_         = 5,
                         int64_t d_r_       = 300,
                         int64_t d_msg_     = 64,
                         int64_t d_dist_    = 2,
                         int64_t n_heads_   = 10,
                         int64_t n_actions_ = N_ACTIONS)
        : n(n_), d_r(d_r_), d_msg(d_msg_), d_dist(d_dist_), n_actions(n_actions_) {

        // ---- PE tables & learnable tokens ----
        sender_pe = register_parameter("sender_pe",
                        torch::randn({n, d_msg}) * 0.02);
        rt_pe     = register_parameter("rt_pe",
                        torch::randn({d_r}) * 0.02);
        tok_value = register_parameter("tok_value",
                        torch::randn({d_r}) * 0.02);

        // ---- Projections & heads ----
        msg_proj     = register_module("msg_proj",
                            torch::nn::Linear(d_msg + d_dist, d_r));
        msg_out_head = register_module("msg_out_head",
                            torch::nn::Linear(d_r, d_msg));
        value_head   = register_module("value_head",
                            torch::nn::Linear(d_r, 1));
        policy_head  = register_module("policy_head",
                            torch::nn::Linear(d_r, n_actions));

        comm_tf = register_module("comm_tf", PreLNAttnBlock(d_r, n_heads_));

        // ---- Precompute other_idx [n, n-1] ----
        std::vector<int64_t> oi_flat;
        oi_flat.reserve(n * (n - 1));
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                if (i != j) oi_flat.push_back(j);
        other_idx = torch::tensor(oi_flat, torch::kLong).view({n, n - 1});
    }

    // ----------------------------------------------------------------
    // Main forward
    //   r_t        : [B, n, d_r]      — from PlayerPolicyNet::forward_r_t
    //   prev_msgs  : [B, n, n, d_msg] — prev_msgs[b,i,j] = m_{j→i}^{t-1}
    //   distances  : [B, n, n, d_dist]— distances[b,i,j] = p_j - p_i
    // ----------------------------------------------------------------
    Output forward(torch::Tensor r_t,
                   torch::Tensor prev_msgs,
                   torch::Tensor distances) {
        namespace I = torch::indexing;
        auto B      = r_t.size(0);
        auto device = r_t.device();
        auto oi     = other_idx.to(device);

        // ---- 1. Gather incoming messages (n-1 per agent, j != i) ----
        auto oi_exp_msg = oi.view({1, n, n-1, 1}).expand({B, n, n-1, d_msg});
        auto incoming   = prev_msgs.gather(2, oi_exp_msg);         // [B, n, n-1, d_msg]

        auto oi_exp_dist = oi.view({1, n, n-1, 1}).expand({B, n, n-1, d_dist});
        auto dist_in     = distances.gather(2, oi_exp_dist);       // [B, n, n-1, d_dist]

        // ---- 2. Add sender PE to RAW messages ----
        auto sp_used = sender_pe.index_select(0, oi.flatten(0, 1))
                                  .view({n, n-1, d_msg});          // [n, n-1, d_msg]
        auto msg_with_pe = incoming + sp_used.unsqueeze(0);        // [B, n, n-1, d_msg]

        // ---- 3. Concat dist → Linear → d_r ----
        auto msg_feat = torch::cat({msg_with_pe, dist_in}, -1);
        auto msg_emb  = msg_proj->forward(msg_feat);               // [B, n, n-1, d_r]

        // ---- 4. r_t: only its dedicated PE, no projection ----
        auto r_emb = r_t + rt_pe.view({1, 1, d_r});                // [B, n, d_r]

        // ---- 5. tok_value: learnable, broadcast to [B, n, d_r] ----
        auto val_emb = tok_value.view({1, 1, d_r})
                              .expand({B, n, d_r});                // [B, n, d_r]

        // ---- 6. Assemble sequence: [r_t, tok_value, m_1..m_{n-1}] ----
        auto seq = torch::cat({r_emb.unsqueeze(2),                 // [B, n, 1, d_r]
                               val_emb.unsqueeze(2),               // [B, n, 1, d_r]
                               msg_emb}, 2);                       // [B, n, n-1, d_r]
        // seq: [B, n, n+1, d_r]

        // ---- 7. Batched transformer over B*n sequences ----
        auto seq_flat = seq.reshape({B * n, n + 1, d_r});
        auto out_flat = comm_tf->forward_self(seq_flat);           // [B*n, n+1, d_r]
        auto out      = out_flat.reshape({B, n, n + 1, d_r});

        // ---- 8. Split outputs ----
        auto r_out   = out.index({I::Slice(), I::Slice(), 0});     // [B, n, d_r]
        auto v_out   = out.index({I::Slice(), I::Slice(), 1});     // [B, n, d_r]
        auto msg_out = out.index({I::Slice(), I::Slice(),
                                  I::Slice(2, n + 1)});            // [B, n, n-1, d_r]

        // ---- 9. Heads ----
        auto logits        = policy_head->forward(r_out);          // [B, n, n_actions]
        auto values        = value_head->forward(v_out).squeeze(-1); // [B, n]
        auto new_msg_other = msg_out_head->forward(msg_out);       // [B, n, n-1, d_msg]

        // ---- 10. Scatter to [B, n, n, d_msg], self-slots zeroed ----
        auto new_msgs    = torch::zeros({B, n, n, d_msg}, new_msg_other.options());
        auto oi_exp_out  = oi.view({1, n, n-1, 1}).expand({B, n, n-1, d_msg});
        new_msgs.scatter_(2, oi_exp_out, new_msg_other);

        return {new_msgs, values, logits};
    }

    // Fresh episode: all messages zero.
    torch::Tensor make_empty_messages(int64_t B, torch::Device device) {
        return torch::zeros({B, n, n, d_msg},
                torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }

    // outgoing[b,i,j] = m_{i→j}   →   incoming[b,j,i] = m_{i→j}
    torch::Tensor outgoing_to_incoming(torch::Tensor outgoing) {
        return outgoing.transpose(1, 2).contiguous();
    }
};
TORCH_MODULE(CommExtensionNet);