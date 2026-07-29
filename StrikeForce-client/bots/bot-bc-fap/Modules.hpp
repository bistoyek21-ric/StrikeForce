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
#include <vector>
#include <set>
#include <map>
#include <numeric>

struct PreLNAttnBlockImpl : torch::nn::Module {
    torch::nn::MultiheadAttention attn{nullptr};
    // norm_q: applied to the query input (self- and cross-attention)
    // norm_kv: applied to the key/value input (cross-attention only; for
    //          self-attention q==kv so norm_q is reused and norm_kv is unused)
    // norm_ffn: applied before the FFN
    torch::nn::LayerNorm norm_q{nullptr}, norm_kv{nullptr}, norm_ffn{nullptr};
    torch::nn::Linear ffn1{nullptr}, ffn2{nullptr};

    PreLNAttnBlockImpl(int64_t d_model, int64_t n_heads, int64_t ffn_mult = 4) {
        // NOTE: no .batch_first(true) here — not a valid option in C++.
        // torch::nn::MultiheadAttention already contains the Q/K/V and output
        // linear projections internally (in_proj_weight/in_proj_bias + out_proj)
        // — no need for separate nn::Linear layers for those.
        attn = register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(d_model, n_heads)));
        norm_q   = register_module("norm_q",   torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        norm_kv  = register_module("norm_kv",  torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        norm_ffn = register_module("norm_ffn", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
        ffn1 = register_module("ffn1", torch::nn::Linear(d_model, ffn_mult * d_model));
        ffn2 = register_module("ffn2", torch::nn::Linear(ffn_mult * d_model, d_model));
    }

    // x: [B, L, D] batch-first in/out; transposed to (L, B, D) only around attn.
    // q == k == v == norm_q(x), so norm_kv is not needed here.
    torch::Tensor forward_self(torch::Tensor x) {
        auto normed = norm_q->forward(x);                    // [B, L, D]
        auto normed_t = normed.transpose(0, 1);              // [L, B, D]
        auto attn_out = std::get<0>(attn->forward(normed_t, normed_t, normed_t)).transpose(0, 1); // [B, L, D]
        x = x + attn_out;
        auto ffn_out = ffn2->forward(torch::relu(ffn1->forward(norm_ffn->forward(x))));
        return x + ffn_out;
    }

    // q: [B, Lq, D], kv: [B, Lk, D] batch-first in/out.
    // q and kv come from different sources/distributions, so each gets its
    // own LayerNorm before attention.
    torch::Tensor forward_cross(torch::Tensor q, torch::Tensor kv) {
        auto q_normed = norm_q->forward(q);
        auto kv_normed = norm_kv->forward(kv);
        auto q_t = q_normed.transpose(0, 1);                  // [Lq, B, D]
        auto kv_t = kv_normed.transpose(0, 1);                // [Lk, B, D]
        auto attn_out = std::get<0>(attn->forward(q_t, kv_t, kv_t)).transpose(0, 1); // [B, Lq, D]
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

    // Separate tables for row-axis vs. column-axis coordinates (genuine 2D
    // positional encoding: axis identity is structural, not a shared lookup).
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

        type_embed      = register_parameter("type_embed", torch::randn({2, d_tok}) * 0.02);
        token_idx_embed = register_parameter("token_idx_embed", torch::randn({K, d_tok}) * 0.02);
        row_coord_embed = register_parameter("row_coord_embed", torch::randn({n, d_tok}) * 0.02);
        col_coord_embed = register_parameter("col_coord_embed", torch::randn({n, d_tok}) * 0.02);
        fusion_block    = register_module("fusion_block", PreLNAttnBlock(d_tok, n_heads));

        out_proj = register_module("out_proj", torch::nn::Linear(d_tok, d_out));
        
        if (in_channels != d_tok)
            proj = register_module("proj", torch::nn::Linear(in_channels, d_tok));
    }

    // seq: [B, n_lines, L, d_tok] (pos-encoded), batched single call
    torch::Tensor axial_summarize(torch::Tensor seq) {
        auto B = seq.size(0), n_lines = seq.size(1), L = seq.size(2);
        auto flat = seq.reshape({B * n_lines, L, d_tok});
        auto q = (queries + query_pe).unsqueeze(0).expand({B * n_lines, K, d_tok});
        auto tok = axial_block->forward_cross(q, flat);
        return tok.reshape({B, n_lines, K, d_tok});
    }

    // Masked mean pool over the token dimension (second-to-last dim of x).
    // mask: 1D [L] (or broadcastable), values in {0,1}. All-ones by default —
    // kept generic so a future variant can null out a token via the mask
    // instead of via a dedicated CLS/output-query token.
    torch::Tensor masked_mean_pool(torch::Tensor x, torch::Tensor mask) {
        auto mask_ = mask.to(x.dtype()).view({1, -1, 1});       // [1, L, 1]
        auto summed = (x * mask_).sum(/*dim=*/1);               // [.., D]
        auto denom = mask_.sum(/*dim=*/1).clamp_min(1e-6);      // [1, 1]
        return summed / denom;
    }

    // Fuse a batch of M arbitrary (row,col) cells given full row_tok [B,n,K,d]
    // and col_tok [B,n,K,d]. row_coord_ids/col_coord_ids are the true
    // coordinate ids (0..n-1) used to look up row_coord_embed/col_coord_embed.
    //
    // F_ij = pool( fusion_layers( [R^i_1..R^i_K, C^j_1..C^j_K] ) )
    // where R^i_k is labelled with (type=row, coord=i, token_idx=k) and
    // C^j_k is labelled with (type=col, coord=j, token_idx=k). No CLS token.
    torch::Tensor fuse_selected(torch::Tensor row_tok, torch::Tensor col_tok,
                                 torch::Tensor row_sel_idx, torch::Tensor col_sel_idx,
                                 torch::Tensor row_coord_ids, torch::Tensor col_coord_ids) {
        auto B = row_tok.size(0);
        auto M = row_sel_idx.size(0);

        auto row_sel = row_tok.index_select(1, row_sel_idx);   // [B, M, K, d_tok]
        auto col_sel = col_tok.index_select(1, col_sel_idx);   // [B, M, K, d_tok]

        auto row_coord = row_coord_embed.index_select(0, row_coord_ids); // [M, d_tok]
        auto col_coord = col_coord_embed.index_select(0, col_coord_ids); // [M, d_tok]

        auto row_label = type_embed[0].view({1,1,d_tok}) + token_idx_embed.view({1,K,d_tok})
                        + row_coord.view({M,1,d_tok});
        auto col_label = type_embed[1].view({1,1,d_tok}) + token_idx_embed.view({1,K,d_tok})
                        + col_coord.view({M,1,d_tok});

        row_sel = row_sel + row_label.unsqueeze(0);   // [B,M,K,d]
        col_sel = col_sel + col_label.unsqueeze(0);

        auto fusion_seq = torch::cat({row_sel, col_sel}, 2);   // [B,M,2K,d] — no CLS
        fusion_seq = fusion_seq.reshape({B * M, 2 * K, d_tok});

        fusion_seq = fusion_block->forward_self(fusion_seq);

        auto mask = torch::ones({2 * K}, fusion_seq.options());
        auto pooled = masked_mean_pool(fusion_seq, mask);       // [B*M, d_tok]
        
        auto z = out_proj->forward(pooled);
        return z.reshape({B, M, d_out});
    }

    struct SparseGrid {
        torch::Tensor values;              // [B, M, d_out]
        torch::Tensor cell_index;          // [n, n] long, -1 if not computed, else position in `values`
        std::vector<int64_t> unique_rows, unique_cols;
    };

    SparseGrid forward_stage1(torch::Tensor x, const std::vector<std::pair<int64_t,int64_t>>& coords) {
        if (proj) x = proj->forward(x.permute({0,2,3,1})).permute({0,3,1,2});

        auto rows = x.permute({0,2,3,1}) + pos1d.view({1,1,n,d_tok});
        auto cols = x.permute({0,3,2,1}) + pos1d.view({1,1,n,d_tok});

        auto combined_seq = torch::cat({rows, cols}, /*dim=*/1);  // [B, 2*n, n, d_tok]
        
        auto combined = axial_summarize(combined_seq);

        auto row_tok = combined.narrow(1, 0, n);        // [B, n, K, d_tok]
        auto col_tok = combined.narrow(1, n, n);        // [B, n, K, d_tok]

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

        auto ridx_t = torch::tensor(rows_idx, torch::kLong);
        auto cidx_t = torch::tensor(cols_idx, torch::kLong);
        auto values = fuse_selected(row_tok, col_tok, ridx_t, cidx_t, ridx_t, cidx_t);

        auto cell_index = torch::full({n, n}, -1, torch::kLong);
        for (size_t k = 0; k < rows_idx.size(); ++k)
            cell_index[rows_idx[k]][cols_idx[k]] = (int64_t)k;

        return {values, cell_index, unique_rows, unique_cols};
    }

    torch::Tensor forward_stage2(const SparseGrid& g, const std::vector<std::pair<int64_t,int64_t>>& coords) {
        auto B = g.values.size(0);
        torch::Tensor v = g.values;                       // [B, M, C_in]
        if (proj) v = proj->forward(v);                   // [B, M, d_tok]

        auto row_ids_t = torch::tensor(g.unique_rows, torch::kLong);
        auto col_ids_t = torch::tensor(g.unique_cols, torch::kLong);

        auto row_gather = g.cell_index.index_select(0, row_ids_t);     // [r, n]
        auto col_gather = g.cell_index.index_select(1, col_ids_t).transpose(0, 1); // [c, n]

        auto row_seq = v.index({torch::indexing::Slice(), row_gather}); // [B, r, n, d_tok]
        auto col_seq = v.index({torch::indexing::Slice(), col_gather}); // [B, c, n, d_tok]

        row_seq = row_seq + pos1d.view({1,1,n,d_tok});
        col_seq = col_seq + pos1d.view({1,1,n,d_tok});

        auto combined_seq = torch::cat({row_seq, col_seq}, /*dim=*/1);  // [B, r+c, n, d_tok]
        
        auto combined = axial_summarize(combined_seq);

        auto row_tok = combined.narrow(1, 0, row_seq.size(1));               // [B, r, K, d_tok]
        auto col_tok = combined.narrow(1, row_seq.size(1), col_seq.size(1)); // [B, c, K, d_tok]

        std::map<int64_t,int64_t> row_pos, col_pos;
        for (size_t k = 0; k < g.unique_rows.size(); ++k) row_pos[g.unique_rows[k]] = (int64_t)k;
        for (size_t k = 0; k < g.unique_cols.size(); ++k) col_pos[g.unique_cols[k]] = (int64_t)k;

        std::vector<int64_t> r_sel, c_sel, r_id, c_id;
        for (auto& pc : coords) {
            r_sel.push_back(row_pos.at(pc.first));
            c_sel.push_back(col_pos.at(pc.second));
            r_id.push_back(pc.first);
            c_id.push_back(pc.second);
        }
        auto r_sel_t = torch::tensor(r_sel, torch::kLong);
        auto c_sel_t = torch::tensor(c_sel, torch::kLong);
        auto r_id_t  = torch::tensor(r_id,  torch::kLong);
        auto c_id_t  = torch::tensor(c_id,  torch::kLong);

        return fuse_selected(row_tok, col_tok, r_sel_t, c_sel_t, r_id_t, c_id_t); // [B, P, d_out]
    }
};
TORCH_MODULE(AFCStageSparse);

struct AFCBackboneSparseImpl : torch::nn::Module {
    AFCStageSparse stage1{nullptr}, stage2{nullptr};

    AFCBackboneSparseImpl(int64_t C = 32, int64_t d_tok1 = 128, int64_t d_out1 = 128,
                int64_t d_tok2 = 256, int64_t d_out2 = 256, int64_t n = 31, int64_t K = 6) {
        stage1 = register_module("stage1", AFCStageSparse(C, d_tok1, d_out1, n, K, 4));
        stage2 = register_module("stage2", AFCStageSparse(d_out1, d_tok2, d_out2, n, K, 8));
    }

    torch::Tensor forward(torch::Tensor x, const std::vector<std::pair<int64_t,int64_t>>& coords
         = {std::pair<int64_t,int64_t>{15, 15}}) {
        auto g1 = stage1->forward_stage1(x, coords);
        return stage2->forward_stage2(g1, coords);   // [B, P, d_out2]
    }
};
TORCH_MODULE(AFCBackboneSparse);

struct PlayerPolicyNetImpl : torch::nn::Module {
    static constexpr int64_t N_ACTIONS = 9;
    static constexpr int64_t FIXED_ROW = 15, FIXED_COL = 15;
    static constexpr int64_t WINDOW_SIZE = 31;
    static constexpr int64_t d_model = 300;

    int64_t n, C;
    AFCBackboneSparse afc{nullptr};

    // Projection 297 -> d_model
    torch::nn::Linear proj_in{nullptr};
    torch::nn::LayerNorm proj_norm{nullptr};

    // Transformer blocks
    PreLNAttnBlock tf_value{nullptr}, tf_policy{nullptr};

    // CLS token and positional embeddings
    torch::Tensor cls_token;   // [1, 1, d_model]
    torch::Tensor pos_embed;   // [1, 1+WINDOW_SIZE, d_model]

    // Final heads
    torch::nn::Linear value_head{nullptr}, policy_head{nullptr};

    // Sliding window buffer
    torch::Tensor buffer;       // [B, WINDOW_SIZE, d_model]
    bool empty;

    PlayerPolicyNetImpl(int64_t C_ = 32, int64_t n_ = 31, int64_t afc_d_out2 = 256)
        : n(n_), C(C_) {

        afc = register_module("afc", AFCBackboneSparse(
            C, 128, 128, 256, afc_d_out2, n, 6));

        int64_t concat_dim = afc_d_out2 + C + N_ACTIONS;  // 297
        proj_in  = register_module("proj_in",  torch::nn::Linear(concat_dim, d_model));
        proj_norm = register_module("proj_norm",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));

        // Two transformer layers
        tf_value = register_module("tf_value", PreLNAttnBlock(d_model, 10));
        tf_policy = register_module("tf_policy", PreLNAttnBlock(d_model, 10));

        // CLS token (content, no position yet)
        cls_token = register_parameter("cls_token", torch::randn({1, 1, d_model}) * 0.02);
        // Position embedding for the whole sequence (CLS + 31 frames)
        pos_embed = register_parameter("pos_embed",
            torch::randn({1, 1 + WINDOW_SIZE, d_model}) * 0.02);
        
        value_head  = register_module("value_head",  torch::nn::Linear(d_model, 1));
        policy_head = register_module("policy_head", torch::nn::Linear(d_model, N_ACTIONS));

        reset_memory();
    }

    void reset_memory() {
        empty = true;

        buffer = torch::zeros({1}, torch::TensorOptions()
            .device(proj_in->weight.device()));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(
        torch::Tensor x, torch::Tensor prev_action) {

        namespace I = torch::indexing;
        auto B = x.size(0);
        if (!empty && buffer.size(0) != B)
            reset_memory();

        // --- Build current timestep feature ---
        auto afc_out = afc->forward(x);                     // [B, 1, 256]
        auto A_vec   = afc_out.squeeze(1);                  // [B, 256]
        auto B_vec   = x.index({I::Slice(), I::Slice(),
                                FIXED_ROW, FIXED_COL});    // [B, C]
        auto C_vec   = torch::one_hot(prev_action, N_ACTIONS).to(A_vec.dtype());
        auto x_cat   = torch::cat({A_vec, B_vec, C_vec}, 1);// [B, 297]
        auto x_proj  = proj_norm->forward(proj_in->forward(x_cat)); // [B, d_model]

        // --- Update sliding buffer (full 31-step BPTT) ---
        if(empty) {
            buffer = x_proj.unsqueeze(1);  // [B, 1, d_model]
            empty = false;
        }
        else if(buffer.size(1) < WINDOW_SIZE)
            buffer = torch::cat({
                buffer,
                x_proj.unsqueeze(1)
            }, 1);   // [B, len + 1 (len < WINDOW_SIZE), d_model]
        else
            buffer = torch::cat({
                buffer.index({I::Slice(), I::Slice(1, WINDOW_SIZE)}),
                x_proj.unsqueeze(1)
            }, 1);   // [B, WINDOW_SIZE, d_model]

        // --- Prepend CLS token and add positional embeddings ---
        auto cls = cls_token.expand({B, 1, d_model});           // [B, 1, d_model]
        auto seq = torch::cat({cls, buffer}, 1);                // [B, 1+WINDOW_SIZE, d_model]
        seq = seq + pos_embed.index({I::Slice(), I::Slice(0, buffer.size(1) + 1),
                            I::Slice()});                                  // add position

        auto out = tf_value->forward_self(seq); // value transformer

        auto cls_out = out.index({I::Slice(), 0});              // [B, d_model]
        auto value  = value_head->forward(cls_out);             // [B, 1]

        out = tf_policy->forward_self(seq);    // policy transformer

        cls_out = out.index({I::Slice(), 0});                   // [B, d_model]
        auto logits = policy_head->forward(cls_out);            // [B, 9]

        //buffer = buffer.detach();

        return {value, logits};
    }
};
TORCH_MODULE(PlayerPolicyNet);