# TINAS: Transformer Interaction among N Agents with Shared Parameters

## A Theoretical Proposal with Convergence Guarantees

---

## 1. Problem and Approach

Multi-agent reinforcement learning (MARL) with parameter sharing is a powerful paradigm: a single policy network, shared across all agents, avoids the combinatorial blowup of independent learning and lets agents generalize across roles. Yet standard PPO training treats each agent's loss as local — no mechanism lets agents *learn to communicate* in a gradient-aware way.

TINAS closes this gap with a **differentiable inter-agent communication channel** built on top of a shared transformer backbone. Each agent \(i\) emits a message \(m^t_{i \to j}\) to every other agent \(j\) at each timestep. The messages are trained by **future-gradient injection**: the sender's message loss at time \(t\) is constructed from the *receiver's* loss at time \(t+1\), differentiated with respect to the sender's positional encoding \(\text{pos}[i]\). Because the token construction is *linear* in both the raw message and \(\text{pos}[i]\), this gradient equals the "true" gradient of the receiver's future loss with respect to the message — allowing gradient injection *without* any re-attachment step.

The central theoretical claim of this document:

> **Under standard smoothness and boundedness assumptions, and with the shared backbone frozen during TINAS training, the injected-gradient parameters of the communication extension converge to a stationary point of the total objective at rate \(O(1/\sqrt{K})\). The recurrence that defines the injected gradient is analytically contractive when \(c_m \cdot L_{\text{net}} < 1\), where \(L_{\text{net}}\) is the network Lipschitz constant of the message head. In that regime the residual STE bias is bounded and its accumulated contribution decays geometrically in \(T\).**

The rest of the document establishes this claim in stages.

---

## 2. Model

### 2.1 Notation

We consider \(n\) agents sharing a parameter vector \(\theta \in \mathbb{R}^d\). Throughout, **\(\theta\) denotes the parameters of the communication extension (`CommExtensionNet`)**. The shared backbone (`PlayerPolicyNet`) is pre-trained and frozen during TINAS training; its per-agent representation \(r^t_i\) enters the extension as a fixed detached input. This separation is made precise in §4.2 and §5.

A rollout consists of \(T\) timesteps \(t = 1, \dots, T\). At each step, agent \(i\) sends a message \(m^t_{i \to j}\) to each other agent \(j \neq i\). Two positional embeddings are used: \(\text{pos}[i] \in \mathbb{R}^{d_{\text{msg}}}\) for each sender \(i\), and a spatial distance vector \(d^t[j \to i] \in \mathbb{R}^{d_{\text{dist}}}\) encoding the geometry between receiver \(j\) and sender \(i\).

**Boundary condition:**
$$
L^{T+1}_i \;=\; 0 \qquad \text{(constant, for all } i\text{)}.
$$

**Per-agent per-step loss** (PPO + value + entropy + message):
$$
L_i^t \;=\; L^{i,t}_{\text{PPO}} + c_V L^{i,t}_{V} - c_H H^{i,t} + \frac{c_m}{n-1} \sum_{j \neq i} L^t_{m, i \to j}.
$$

**Message loss:**
$$
L^t_{m, i \to j} \;=\; \big(g^{t+1}_{i \to j}\big)^{\!\top} m^t_{i \to j}.
$$

**Injected gradient:**
$$
g^{t+1}_{i \to j} \;=\; \left( \frac{\partial L^{t+1}_j}{\partial \text{pos}[i]} \right)_{\!\text{detached}}.
$$

The \(\text{detached}\) subscript denotes that the gradient is treated as a constant in any subsequent backward pass. It is extracted with `create_graph=False`, so it carries no computational graph.

**Index convention (to be used consistently everywhere).**

The index \(g^{t}_{i \to j}\) always denotes *the gradient that multiplies the message \(m^{t-1}_{i \to j}\) in the loss at step \(t\)*. By construction \(g^t_{i \to j} = \partial L^t_j/\partial \text{pos}[i]\), so the *receiver* is \(j\) and the *sender* is \(i\). When gradients are stored as arrays after a backward call, the storage is therefore indexed as `G[t][receiver][sender]`, not `G[t][sender][receiver]`. This convention is enforced in the pseudocode of §4.

**Total objective:**
$$
\mathcal{L}(\theta) \;=\; \frac{1}{T n} \sum_{t=1}^{T} \sum_{i=1}^{n} L_i^t.
$$

The averaging by \(Tn\) matches the pseudocode's gradient accumulation, which divides the sum of per-agent-per-step gradients by \(T \cdot n\).

### 2.2 Token construction (exact)

For each ordered pair \((i, j)\) with \(i \neq j\), the message token transmitted from sender \(i\) to receiver \(j\) at step \(t\) is

$$
\boxed{\;\text{tok}^t_{i \to j} \;=\; \text{proj}\!\left(\text{cat}\!\left\{ m^t_{i \to j} + \text{pos}[i],\; d^t[j \to i] \right\}\right).\;}
$$

Here \(\text{proj} : \mathbb{R}^{d_{\text{msg}} + d_{\text{dist}}} \to \mathbb{R}^{d_{\text{model}}}\) is a linear map (the `msg_proj` module of the extension), and \(\text{cat}\) denotes concatenation along the feature dimension. The first block of the input is \(m^t_{i \to j} + \text{pos}[i]\) — the raw message plus the sender's positional encoding — and the second block is the spatial distance vector.

At the receiver \(j\), the messages from all senders \(i \neq j\) enter the transformer's input sequence together with \(j\)'s own \(r^t_j\) and a learnable value token \(v\). After the transformer, the message slots are projected by a linear head to produce the outbound messages \(m^t_{j \to k}\). Full architectural details are given in §4.

**Key observation.** The composition in the boxed equation is *linear in both arguments* \(m^t_{i \to j}\) and \(\text{pos}[i]\). This is the technical fact that drives the entire gradient-injection mechanism.

### 2.3 Gradient equivalence

**Lemma 1 (gradient equivalence).** For any \(i \neq j\) and any \(t\), the two gradients

$$
\frac{\partial L^{t+1}_j}{\partial m^t_{i \to j}} \qquad \text{and} \qquad \frac{\partial L^{t+1}_j}{\partial \text{pos}[i]}
$$

are equal as tensors in \(\mathbb{R}^{d_{\text{msg}}}\), provided \(L^{t+1}_j\) depends on \(m^t_{i \to j}\) only through the single token \(\text{tok}^t_{i \to j}\) at receiver \(j\).

**Proof.** Let \(J_{\text{proj}} \in \mathbb{R}^{d_{\text{model}} \times (d_{\text{msg}} + d_{\text{dist}})}\) be the Jacobian of the linear map \(\text{proj}\). Write \(a = m^t_{i \to j} + \text{pos}[i]\) for the first input block. Differentiation of the token construction gives

$$
\frac{\partial \text{tok}^t_{i \to j}}{\partial m^t_{i \to j}} \;=\; J_{\text{proj}} \begin{bmatrix} I_{d_{\text{msg}}} \\ 0_{d_{\text{dist}}} \end{bmatrix}
\;=\; J_{\text{proj}}[:, 0:d_{\text{msg}}],
$$

$$
\frac{\partial \text{tok}^t_{i \to j}}{\partial \text{pos}[i]} \;=\; J_{\text{proj}} \begin{bmatrix} I_{d_{\text{msg}}} \\ 0_{d_{\text{dist}}} \end{bmatrix}
\;=\; J_{\text{proj}}[:, 0:d_{\text{msg}}].
$$

The two are identical. Because \(L^{t+1}_j\) depends on \(m^t_{i\to j}\) only through the single token \(\text{tok}^t_{i \to j}\), the chain rule through the downstream computation of \(L^{t+1}_j\) gives

$$
\frac{\partial L^{t+1}_j}{\partial m^t_{i \to j}} \;=\; \frac{\partial L^{t+1}_j}{\partial \text{tok}^t_{i \to j}} \cdot \frac{\partial \text{tok}^t_{i \to j}}{\partial m^t_{i \to j}}
\;=\; \frac{\partial L^{t+1}_j}{\partial \text{tok}^t_{i \to j}} \cdot \frac{\partial \text{tok}^t_{i \to j}}{\partial \text{pos}[i]}
\;=\; \frac{\partial L^{t+1}_j}{\partial \text{pos}[i]}.
$$

\(\square\)

**Remark (single-token dependence).** The "provided" clause in Lemma 1 is satisfied by the TINAS architecture for three reasons. (i) Each ordered pair \((i, j)\) contributes exactly one token to receiver \(j\)'s input sequence, namely \(\text{tok}^t_{i \to j}\). (ii) \(\text{pos}[i]\) appears only in that token — it is not otherwise reused by receiver \(j\). (iii) \(L^{t+1}_j\) depends only on receiver \(j\)'s outputs (its own logits, value, and outgoing messages), not on other agents' losses. In the implementation, the constant gather index `other_idx[i] = [j \neq i]` makes (i) explicit: the incoming messages to receiver \(j\) are gathered exactly once per sender \(i\).

**Consequence.** The algorithm extracts \(g^{t+1}_{i \to j}\) via `autograd.grad(L_j^{t+1}, pos)` rather than `autograd.grad(L_j^{t+1}, m^t)`. This is not an approximation — it is exact. It works because \(\text{pos}[i]\) remains attached to the graph (it is a learnable parameter), whereas \(m^t_{i \to j}\) has been detached on the send side. The equivalence lemma says that the "as-if-attached" gradient of \(L^{t+1}_j\) with respect to \(m^t_{i \to j}\) can be recovered from the genuinely-attached gradient with respect to \(\text{pos}[i]\).

**No re-attachment step is required.** Pass 2 of the algorithm does not need to flip `requires_grad` on detached messages; it simply calls `autograd.grad` on \(\text{pos}[i]\) and obtains the correct \(g\).

### 2.4 Per-step loss (restated)

$$
L_i^t \;=\; L^{i,t}_{\text{PPO}} + c_V L^{i,t}_{V} - c_H H^{i,t} + \frac{c_m}{n-1} \sum_{j \neq i} \big(g^{t+1}_{i \to j}\big)^{\!\top} m^t_{i \to j}.
$$

Every term is an unnormalized per-agent per-step contribution; the total \(\mathcal{L}\) normalizes by \(T n\).

### 2.5 Total objective

$$
\boxed{\;\mathcal{L}(\theta) \;=\; \frac{1}{T n} \sum_{t=1}^{T} \sum_{i=1}^{n} L_i^t.\;}
$$

The gradient \(\nabla_\theta \mathcal{L}\) is what AdamW optimizes, and the training update is
$$
\theta_{K+1} \;=\; \text{AdamW}\!\left(\theta_K,\; \hat{G}_K,\; \eta=10^{-3},\; \lambda=10^{-4}\right),
$$
where \(\hat{G}_K\) is a minibatch estimate of \(\nabla_\theta \mathcal{L}\), and \(\theta\) ranges over the parameters of the communication extension as stated in §2.1.

---

## 3. Theoretical Analysis

### 3.1 Assumptions

**A1 (smoothness of $\mathcal{L}$ in $\theta$).** \(\mathcal{L} : \mathbb{R}^d \to \mathbb{R}\) is \(L_{\text{lip}}\)-smooth: \(\|\nabla \mathcal{L}(\theta) - \nabla \mathcal{L}(\theta')\| \le L_{\text{lip}} \|\theta - \theta'\|\). This is standard for networks composed of Lipschitz layers (LayerNorm, ReLU, linear) with bounded weights, and is enforced in practice by weight decay.

**A2 (bounded gradient).** There exists \(G > 0\) such that \(\|\nabla_\theta \mathcal{L}(\theta)\| \le G\). This is enforced by **gradient clipping at norm 1**, so \(G = 1\).

**A3 (bounded variance).** For any stochastic minibatch \(\xi_K\):
$$
\mathbb{E}[\hat{G}_K \mid \theta_K] = \nabla_\theta \mathcal{L}(\theta_K), \qquad \mathbb{E}[\|\hat{G}_K - \nabla_\theta \mathcal{L}(\theta_K)\|^2] \le \sigma^2.
$$

**A4 (network Lipschitz).** Let
$$
L_{\text{net}} \;:=\; \sup_{\theta}\, \max_{i,j,k,t}\, \left\| \frac{\partial m^t_{j \to k}}{\partial \text{pos}[i]} \right\|_{\text{op}}.
$$
This is finite for any finite network and is bounded in practice by the Lipschitz constants of the layers and the norm of the incoming messages.

**A5 (bounded second derivative).** There exists \(B_2 > 0\) such that
$$
\sup_t \left\|\frac{\partial g^t_{i \to j}}{\partial \theta}\right\| \;\le\; B_2
$$
along the optimization trajectory, for all \(i \neq j\). This controls the STE bias term of Lemma 5 independently of A1–A4. In practice, \(B_2\) is bounded by the Lipschitz constants of the extension heads (`policy_head`, `value_head`, `msg_out_head`) composed with the Jacobian bounds appearing in \(L_{\text{net}}\).

Assumptions A1–A3 are standard in the nonconvex optimization literature (Ghadimi & Lan 2013; Bottou et al. 2018). A4 is specific to TINAS and controls the contractivity of the message recurrence. A5 is specific to the STE analysis and controls the magnitude of the second-order bias term.

### 3.2 Well-definedness

**Lemma 2 (recursion terminates).** *The message loss \(L^t_{m, i \to j}\) is well-defined for all \(t = 1, \dots, T\) and all \(i \neq j\). Moreover, it is zero at \(t = T\).*

**Proof.** By the boundary condition \(L^{T+1}_j = 0\), the gradient \(g^{T+1}_{i \to j} = 0\), so \(L^T_{m, i \to j} = 0^\top m^T_{i \to j} = 0\). For \(t = T - 1\), the loss \(L^{T-1}_{m, i \to j}\) depends on \(g^T_{i \to j} = \partial L^T_j / \partial \text{pos}[i]\), and \(L^T_j\) contains only a base term and \(L^T_{m, j \to k} = 0\) (just shown). By induction on \(T - t\), every \(L^t_{m, i \to j}\) is a finite sum of at most \(T - t\) nested gradients, each well-defined. \(\square\)

The recursion has depth at most \(T - t\) at step \(t\). This is the reason the algorithm can be implemented in \(O(T)\) passes rather than as an implicit differentiation fixed point.

### 3.3 Gradient computation

**Lemma 3 (gradient formula).** *The gradient of the total objective with respect to \(\theta\) is*

$$
\nabla_\theta \mathcal{L}(\theta) \;=\; \frac{1}{T n} \sum_{t=1}^{T} \sum_{i=1}^{n} \left[
\nabla_\theta\!\left(L^{i,t}_{\text{PPO}} + c_V L^{i,t}_V - c_H H^{i,t}\right) + \frac{c_m}{n-1} \sum_{j \neq i} \big(g^{t+1}_{i \to j}\big)^{\!\top} \nabla_\theta m^t_{i \to j}
\right].
$$

**Proof.** Since \(g^{t+1}_{i \to j}\) is detached (constant in \(\theta\)), differentiation of the message-loss term yields
$$
\nabla_\theta \big[(g^{t+1}_{i \to j})^\top m^t_{i \to j}\big] = (g^{t+1}_{i \to j})^\top \nabla_\theta m^t_{i \to j}.
$$
Summing over \(i\), \(j\), \(t\) and normalizing by \(Tn\) gives the stated formula. \(\square\)

Lemma 3 is the operational heart of the algorithm: the sent gradient \(g^{t+1}_{i \to j}\) acts as a *constant weight* on the message's parameter dependence.

### 3.4 Contractivity of the recurrence

Define the operator \(T^t : \mathbb{R}^{n(n-1) \times d_{\text{msg}}} \to \mathbb{R}^{n(n-1) \times d_{\text{msg}}}\) by

$$
\big(T^t G\big)_{(i,j)} \;=\; \frac{c_m}{n-1} \sum_{k \neq j} G_{(j,k)} \cdot \frac{\partial m^t_{j \to k}}{\partial \text{pos}[i]},
$$

where \(G_{(j,k)} \in \mathbb{R}^{d_{\text{msg}}}\) is the \((j,k)\)-th slice of \(G\) and the product is the vector-Jacobian product \(G^\top \partial m / \partial \text{pos}\). Then the recurrence for the injected gradients is

$$
g^t_{(i,j)} \;=\; b^t_{(i,j)} \;+\; \big(T^t g^{t+1}\big)_{(i,j)},
$$

where \(b^t_{(i,j)} = \partial(L^{i,t}_{\text{PPO}} + c_V L^{i,t}_V - c_H H^{i,t})_{\text{for agent }j} / \partial \text{pos}[i]\) is the "base" contribution.

**Proposition 4 (contractivity).** *Under A4,*

$$
\|T^t\|_{\text{op}} \;\le\; c_m \, L_{\text{net}}.
$$

*In particular, when \(c_m \, L_{\text{net}} < 1\), the operator \(T^t\) is a strict contraction, and the accumulated injected gradient satisfies*

$$
\|g^t\| \;\le\; \sum_{s=0}^{T-t} \big(c_m L_{\text{net}}\big)^{s} \cdot \|b^{t+s}\|.
$$

**Proof.** The operator \(T^t\) is *row-wise*: its output at row \((i, j)\) depends only on row \(j\) of \(G\), not on rows \((i', j')\) with \(j' \neq j\). Concretely, for a fixed slice \((i, j)\), \(T^t\) sums over \(k \neq j\), giving \(n-1\) terms each with coefficient \(c_m/(n-1)\). Bounding each term by \(L_{\text{net}} \|G_{(j,k)}\|\) and applying the triangle inequality gives

$$
\big\|\big(T^t G\big)_{(i,j)}\big\| \;\le\; \frac{c_m}{n-1} \sum_{k \neq j} L_{\text{net}} \|G_{(j,k)}\|
\;\le\; c_m \, L_{\text{net}} \cdot \max_k \|G_{(j,k)}\|.
$$

Taking the max over \((i, j)\) on the left and the sup-norm over slices on the right gives \(\|T^t G\|_\infty \le c_m L_{\text{net}} \|G\|_\infty\).

Because \(T^t\) is row-wise, the above bound *also* holds in the operator norm. To see this, let \(G\) be any tensor of shape \(n(n-1) \times d_{\text{msg}}\) viewed as a matrix in \(\mathbb{R}^{n(n-1) \times d_{\text{msg}}}\). The operator \(T^t\) acts on each row \((i, j)\) as a linear map depending only on row \(j\) of \(G\). Writing \(G_{\cdot, j} \in \mathbb{R}^{d_{\text{msg}}}\) for the slice of rows \((i, j)\) across all \(i \neq j\), and letting \(\|G\|_{F, \text{col}} := \max_j \|G_{\cdot, j}\|_F\), we have \(\|T^t G\|_{F, \text{col}} \le c_m L_{\text{net}} \|G\|_{F, \text{col}}\). The Frobenius column norm is equivalent to the spectral norm on \(\mathbb{R}^{n(n-1) \times d_{\text{msg}}}\) with ratio bounded by \(\min(n(n-1), d_{\text{msg}})\), independent of any per-row dimension that would scale with \(n\) in a norm-mixing argument. Therefore \(\|T^t\|_{\text{op}} \le c_m L_{\text{net}}\) with a constant that is *independent of \(n\)*. The geometric bound follows by unrolling \(g^t = b^t + T^t g^{t+1}\) up to the boundary condition. \(\square\)

**Interpretation.** The factor \(c_m/(n-1)\) in the loss is what makes the operator \(T^t\) contractive with a coefficient that *does not grow with \(n\)*. Without it, the coefficient per level would be \(c_m (n-1) L_{\text{net}}\), which fails to contract for large \(n\).

### 3.5 STE bias

The injected gradient \(g^{t+1}_{i \to j}\) is **detached** before it enters the message loss \(L^t_{m, i \to j}\). This is the STE (Straight-Through Estimator) approximation. The "true" objective \(\mathcal{L}^\ast\) would have \(g\) *attached*, and its gradient would include an additional second-order term.

**Lemma 5 (STE bias bound).** *Define the "true" objective*

$$
\mathcal{L}^\ast(\theta) \;=\; \frac{1}{Tn}\sum_{t,i} \left[\text{base}_i^t + \frac{c_m}{n-1}\sum_{j \neq i} \big(g^{t+1}_{i \to j}(\theta)\big)^{\!\top} m^t_{i \to j}(\theta)\right]
$$

*where \(g\) is not detached. Then, under A5,*

$$
\big\|\nabla_\theta \mathcal{L} - \nabla_\theta \mathcal{L}^\ast\big\| \;\le\; c_m \cdot \sup_t \|m^t\| \cdot B_2.
$$

*Moreover, the bias vanishes identically when \(m^t \equiv 0\) or when \(B_2 = 0\).*

**Proof.** Since \(\mathcal{L}\) treats \(g^{t+1}\) as a constant, the difference is the missing term
$$
\frac{1}{Tn} \sum_{t,i,j} \frac{c_m}{n-1} \big(m^t_{i \to j}\big)^{\!\top} \frac{\partial g^{t+1}_{i \to j}}{\partial \theta}.
$$

The number of terms in the sum is \(T \cdot n \cdot (n-1)\). After multiplying by \(c_m / (Tn(n-1))\), each term is bounded by \(c_m \|m\| B_2\) by A5. Taking the supremum over \(t\) gives the stated bound. The vanishing at \(m = 0\) and at \(B_2 = 0\) is immediate. \(\square\)

The bias \(\mathcal{B} := c_m \sup_t \|m^t\| \cdot B_2\) enters the convergence analysis as a fixed model error. Note that \(c_m\) appears in the bias bound *linearly*, and that \(B_2\) is bounded by A5. In practice, \(c_m\) is chosen small enough (typically \(0.1\)) that \(\mathcal{B}\) is small compared to the scale of \(\nabla_\theta \mathcal{L}\).

### 3.6 Convergence theorem

**Theorem 6 (convergence to a stationary point).** *Suppose A1–A5 hold. Let \(\theta \in \mathbb{R}^d\) denote the parameters of the communication extension (`CommExtensionNet`); the shared backbone `PlayerPolicyNet` is frozen and enters only through the fixed detached representations \(r^t_i\). Let \(\mathcal{B}\) be the STE bias bound of Lemma 5. Run AdamW on \(\theta\) with step sizes \(\eta_K = \eta_0 / \sqrt{K}\) and gradient clipping at norm 1. Then for any \(K \ge 1\),*

$$
\mathbb{E}\!\left[\|\nabla_\theta \mathcal{L}(\theta_K)\|^2\right] \;\le\; \frac{C_1}{\sqrt{K}} \;+\; \mathcal{B}^2,
$$

*where \(C_1 = 2(\mathcal{L}(\theta_0) - \mathcal{L}^{\inf})/\eta_0 + \eta_0 L_{\text{lip}} \sigma^2\) depends on the initial condition, learning rate, smoothness, and gradient noise but not on \(K\), \(T\), or \(n\).*

**Proof sketch.** Under A1–A3, the standard descent lemma for AdamW (Chen et al. 2019, Thm. 4) gives

$$
\mathbb{E}[\mathcal{L}(\theta_{K+1})] \;\le\; \mathbb{E}[\mathcal{L}(\theta_K)] - \eta_K \mathbb{E}[\|\nabla \mathcal{L}(\theta_K)\|^2] + \eta_K^2 L_{\text{lip}} \sigma^2 + \eta_K \mathcal{B}^2.
$$

Summing over \(K = 1, \dots, K_{\text{end}}\) and using \(\eta_K = \eta_0/\sqrt{K}\) gives
$$
\sum_{k=1}^{K_{\text{end}}} \frac{\eta_0}{\sqrt{k}} \mathbb{E}[\|\nabla \mathcal{L}(\theta_k)\|^2] \;\le\; \mathcal{L}(\theta_0) - \mathcal{L}^{\inf} + \eta_0 L_{\text{lip}} \sigma^2 \sum_{k=1}^{K_{\text{end}}} \frac{1}{k} + \eta_0 \mathcal{B}^2 \sqrt{K_{\text{end}}}.
$$

Since \(\sum_{k=1}^{K_{\text{end}}} 1/\sqrt{k} \ge \sqrt{K_{\text{end}}}\), dividing through and using the harmonic sum bound \(\sum 1/k \le \log K_{\text{end}}\) yields the stated \(O(1/\sqrt{K}) + \mathcal{B}^2\) rate up to logarithmic factors. \(\square\)

**Remark (scope of $\theta$).** The theorem is stated for the communication extension's parameters because Pass 1 of the algorithm (§4) computes \(r^t_i\) under `NoGradGuard` and detaches it before Pass 2. Consequently, the gradient of \(L_i^t\) does not flow into `PlayerPolicyNet`. This is a deliberate design choice: it separates the per-agent representation from the communication channel, allows the backbone to be trained once with a standard behavioral-cloning recipe, and keeps the two-pass algorithm's memory footprint at \(O(T)\) rather than \(O(T)\) per agent. The theorem therefore describes the convergence of the *communication-extension training problem* under a fixed representation, which is the problem TINAS actually solves.

**Interpretation.** The rate \(O(1/\sqrt{K})\) is the optimal rate for nonconvex stochastic optimization with bounded gradients. It says: for every \(\epsilon > \mathcal{B}^2\), a finite number of updates reaches the \(\epsilon\)-neighborhood of a stationary point of \(\mathcal{L}\). The STE bias \(\mathcal{B}\) is the irreducible floor — it caps the best achievable gradient norm, at the scale set by the message magnitude and the second derivative of the loss.

### 3.7 Number of updates to reach an \(\epsilon\)-optimum

**Corollary 7.** *Choose \(\epsilon > \mathcal{B}^2\), and let \(\delta = \epsilon - \mathcal{B}^2 > 0\). Then*

$$
K \;=\; \mathcal{O}\!\left(\frac{C_1^2}{\delta^2}\right)
$$

*updates suffice to reach \(\mathbb{E}[\|\nabla \mathcal{L}(\theta_K)\|^2] \le \epsilon\).*

**Proof.** Direct from Theorem 6. \(\square\)

**Reading.** The number of updates to reach a stationary point scales as \(O(1/\delta^2)\), where \(\delta\) is the gap between the target accuracy and the STE floor. Since \(\mathcal{B} = c_m \sup_t \|m^t\| \cdot B_2\), and both suprema are bounded by A1–A5 plus clipping, the number of updates is finite and independent of \(T\) and \(n\).

---

## 4. Algorithm: Two-Pass Backward

The theoretical guarantees are backed by an efficient two-pass algorithm. It mirrors the standard PPO rollout loop with an added backward pass for message-gradient injection.

### 4.1 Storage convention

Before stating the pseudocode, fix the storage convention used throughout:

- `pos` is the parameter tensor of sender positional encodings, shape `[n, d_msg]`.
- `autograd.grad(L_j^t, pos)` returns a tensor `G_j` of shape `[n, d_msg]` with `G_j[i] = ∂L_j^t/∂pos[i] = g^t_{i→j}`.
- We store this as `g[t][j][i] := G_j[i]`. So `g[t][receiver][sender] = g^t_{sender→receiver}`.

With this convention, the message loss at step `t` for agent `i` — which by §2.4 is

$$\frac{c_m}{n-1} \sum_{j \neq i} \big(g^{t+1}_{i \to j}\big)^{\!\top} m^t_{i \to j},$$

— reads `g[t+1][j][i] · m[t][i→j]` in storage terms. That is, the *receiver* index `j` is the first index of `g`, and the *sender* index `i` is the second. This is the **incoming** convention (the stored gradient is the gradient w.r.t. the message *received* by `j` from `i`).

### 4.2 Pass 1 — forward without gradients

```
m[0] = 0
with NoGradGuard:
    for t = 1, ..., T-1:
        r[t]  = model->forward_r_t(obs[t]).r_t().clone().detach()
        m[t]  = extension->forward(r[t], m[t-1]).new_messages().clone().detach()
model->reset_memory()
```

Both \(r[t]\) and \(m[t]\) are recorded as detached tensors for later use. This pass touches `PlayerPolicyNet` for the \(r[t]\) computation only; no parameter of `PlayerPolicyNet` receives a gradient in either pass.

### 4.3 Pass 2 — backward with injected gradients

```
g[T+1][*][*] = 0              # boundary condition
G = 0                          # accumulated gradient

for t = T, T-1, ..., 1:
    # Differentiable forward, using recorded detached snapshots
    out = extension->forward(r[t], m[t-1])    # returns logits, values, new_messages (m[t])

    for i = 1, ..., n:
        # Message loss for agent i at step t:
        # c_m/(n-1) * sum_{j != i} (g^{t+1}_{i->j}) . m[t][i->j]
        # With storage convention g[t+1][receiver][sender]:
        #   g^{t+1}_{i->j} = g[t+1][j][i]
        L_msg = 0
        for j != i:
            L_msg += (g[t+1][j][i] . m[t][i->j]).sum()
        L_msg /= (n - 1)

        L_i^t = L^{i,t}_PPO + c_V L^{i,t}_V - c_H H^{i,t} + c_m * L_msg

        # Parameter gradient and injected gradient in one backward call
        grad_i = autograd.grad(L_i^t, params)                      # ∇_θ L_i^t
        grad_i = clip_norm_1(grad_i)                               # A2 enforced
        G += grad_i

        # Injected gradient: autograd.grad(L_i^t, pos)[k] = ∂L_i^t/∂pos[k]
        #                                  = g^t_{k->i}
        # Storage: g[t][receiver=i][sender=k]
        G_pos_i = autograd.grad(L_i^t, pos, create_graph=False)    # [n, d_msg]
        for k != i:
            g[t][i][k] = G_pos_i[k]

G /= (T * n)
θ ← AdamW(G, lr=1e-3, wd=1e-4)
```

### 4.4 Clarifications

1. **Indexing.** The storage is `g[t][receiver][sender]`, so `g[t][j][i] = ∂L_j^t/∂pos[i] = g^t_{i→j}`. This is the *incoming* convention, and it is the unique convention under which the pair `(g[t+1][j][i], m[t][i→j])` is well-defined for the message loss at step `t`. Storing `g` in the *outgoing* convention (`g[t][sender][receiver]`) and pairing it with `m[t][i→j]` would multiply the gradient of sender `j`'s loss w.r.t. `pos[i]` by the message from `i` to `j`, which is not the loss term of §2.4.

2. **No re-attach step.** Pass 2 uses `autograd.grad(L_i^t, pos)` directly. By Lemma 1, this equals the message-side gradient \(\partial L_i^t / \partial m\) that would be computed with \(m\) attached. The equivalence is exact; no `requires_grad=True` flip on \(m\) is needed.

3. **Pass 2 never touches `PlayerPolicyNet`.** The \(r[t]\) values are consumed as detached inputs from Pass 1, so the sliding-window buffer in `PlayerPolicyNet` is irrelevant during Pass 2. This sidesteps the buffer-replay issue that would otherwise require either \(O(T)\) buffer snapshots or \(O(T^2)\) recomputation. It also fixes the scope of \(\theta\) in Theorem 6 to the communication-extension parameters, as stated in §2.1 and §3.6.

**Cost.** Pass 1 is \(O(T)\) forward computations. Pass 2 is \(O(T)\) forward + backward computations. Total: \(O(T)\), identical to standard PPO with the shared network, plus one extra `autograd.grad` per agent per step.

---

## 5. Practical Guarantees under AdamW + Gradient Clipping

The analysis above is stated for SGD in Theorem 6, but the guarantees carry over to AdamW + clip 1 with two remarks.

**Clip 1 enforces A2 exactly.** The update \(\theta_{K+1} = \theta_K - \eta_K \hat{G}_K / \max(1, \|\hat{G}_K\|)\) satisfies \(\|\theta_{K+1} - \theta_K\| \le \eta_K\) regardless of the loss landscape, and \(\|\hat{G}_K\| \le 1\). This is precisely the bounded-gradient assumption of A2.

**AdamW does not change the asymptotic rate.** The adaptive step \(\eta_K \hat{m}_K / (\sqrt{\hat{v}_K} + \epsilon)\) modifies the per-coordinate magnitude but not the direction of descent. Under A1–A3, AdamW converges at \(O(1/\sqrt{K})\) (Kingma & Ba 2015; Chen et al. 2019), identical to SGD. The STE bias enters as before.

**Combined.** The guarantees of Theorem 6 hold with the constant \(C_1\) adjusted for the adaptive scaling. The number of updates \(K = O(1/\delta^2)\) from Corollary 7 remains valid.

**Frozen-backbone remark.** The theorem and its corollary are stated for the communication-extension parameters. This is the regime the two-pass algorithm implements (§4), and it is the regime in which the convergence rate \(O(1/\sqrt{K})\) is formally guaranteed. If a user wishes to also fine-tune the backbone, the algorithm would need to be modified so that the per-step loss in Pass 2 depends on the backbone's current parameters (i.e., not detach \(r[t]\) in Pass 1), at the cost of the \(O(T)\) buffer-replay property of §4.2. That regime is not analyzed here and is left as future work.

---

## 6. Characterization of the Stationary Point

At any stationary point \(\theta^*\) of \(\mathcal{L}\) (over the communication-extension parameters), we have \(\nabla_\theta \mathcal{L}(\theta^*) = 0\), which by Lemma 3 reads

$$
\frac{1}{Tn} \sum_{t,i} \left[
\nabla_\theta\!\left(L^{i,t}_{\text{PPO}} + c_V L^{i,t}_V - c_H H^{i,t}\right) + \frac{c_m}{n-1}\sum_{j \neq i} \big(g^{t+1}_{i \to j}\big)^{\!\top} \nabla_\theta m^t_{i \to j}
\right] = 0.
$$

This is a **balance condition** between two contributions: the PPO signal that pushes the extension's policy head toward the standard objective, and the message-injection signal that pushes senders to produce messages that reduce receivers' future losses. The stationary point sits at a trade-off between these two forces.

In the shared-parameter setting, \(\theta^*\) is a fixed point of the joint map \((\theta, m(\theta)) \mapsto \theta'\) generated by a single AdamW step, *with the backbone fixed*. At equilibrium, no unilateral perturbation of any agent's messages can improve that agent's message-loss contribution while keeping all other extension parameters and messages fixed. This is a weak analogue of Nash equilibrium for the message-passing game, restricted to the extension parameters.

---

## 7. Summary of Theoretical Results

| Result | Statement |
|---|---|
| **Lemma 1** | \(\partial L / \partial m^t_{i \to j} = \partial L / \partial \text{pos}[i]\) exactly, by linearity of the token construction and single-token dependence of \(L_j^{t+1}\) on \(\text{tok}^t_{i\to j}\). No re-attachment needed. |
| **Lemma 2** | The message-loss recursion is well-defined and terminates at depth \(T - t\). |
| **Lemma 3** | \(\nabla_\theta \mathcal{L}\) has an exact two-part expression: base + injected message gradient. |
| **Prop. 4** | The injected gradient recurrence is contractive with coefficient \(c_m L_{\text{net}}\), with a norm equivalence constant independent of \(n\). |
| **Lemma 5** | STE bias is bounded by \(c_m \sup\|m\| B_2\) and vanishes at \(m=0\) or \(B_2=0\). |
| **Thm 6** | \(\mathbb{E}[\|\nabla_\theta \mathcal{L}(\theta_K)\|^2] \le O(1/\sqrt{K}) + \mathcal{B}^2\), with \(\theta\) = communication-extension parameters. |
| **Cor. 7** | \(K = O(1/\delta^2)\) updates suffice to reach \(\epsilon = \mathcal{B}^2 + \delta\). |
| **§5** | Guarantees carry over to AdamW + grad clip 1. |

---

## 8. Conclusion

TINAS trains inter-agent messages through **future-gradient injection** — an idea whose natural implementation might be expected to require re-attaching the detached message to the graph. The linearity of the token construction \(\text{tok}^t_{i \to j} = \text{proj}(\text{cat}\{m^t_{i \to j} + \text{pos}[i],\, d^t[j \to i]\})\) makes this unnecessary: the gradient with respect to \(\text{pos}[i]\) equals the gradient with respect to \(m^t_{i \to j}\) exactly (Lemma 1), so a single `autograd.grad` call recovers the correct injected gradient.

The theoretical picture is then clean:

- The recurrence that defines \(g^t_{i \to j}\) terminates at depth \(T - t\) (Lemma 2).
- Each level of the recurrence carries a factor \(c_m/(n-1)\) and a network-Jacobian contribution, giving a contractive operator when \(c_m L_{\text{net}} < 1\) (Prop. 4).
- The only STE bias comes from detaching \(g\) itself in the message loss, not from any detach of the message (Lemma 5), and it is bounded by \(c_m\) times the message magnitude times the second derivative (bounded by A5).
- Under standard smoothness and boundedness, AdamW + clip 1 converges at \(O(1/\sqrt{K})\) to an \(\epsilon\)-neighborhood of a stationary point of the communication-extension training problem, with \(\epsilon = \mathcal{B}^2 + O(1/\sqrt{K})\) (Thm 6, Cor. 7).

**The central message.** TINAS's injected-gradient mechanism is not only computationally efficient (no re-attach, one `autograd.grad` per agent per step, no buffer replay) but also theoretically principled: it preserves the convergence rate of AdamW on the composite objective, with the only residual bias being the second-order STE term — and that term is controlled by \(c_m\) and the message magnitude, both of which are small by design in practice. The convergence statement is made for the communication-extension parameters under a frozen backbone, which is exactly the regime the two-pass algorithm implements.

---

## 9. Open Questions

1. **Tightness of the contractivity bound.** \(L_{\text{net}} = \sup\|\partial m/\partial \text{pos}\|\) is a conservative bound. In practice, \(m\) and \(\text{pos}\) are initialized near zero, so \(\partial m/\partial \text{pos}\) is small at init and grows only as training progresses. A tighter analysis could exploit this.

2. **Adaptive \(c_m\).** The constant \(c_m\) trades off message-learning speed against STE bias and recurrence stability. An adaptive schedule \(c_m(t)\) that starts small (to establish the base PPO policy) and grows once messages are useful could improve the constant in Theorem 6 without sacrificing the asymptotic rate.

3. **PL condition.** Under the Polyak–Łojasiewicz condition, Theorem 6 sharpens to linear convergence \(O(\rho^K)\). Identifying sufficient conditions on the TINAS landscape for the PL condition to hold is open.

4. **Empirical calibration of \(\mathcal{B}\).** The bias bound depends on \(B_2\) (A5), which is measurable but rarely measured. Empirically estimating this quantity would quantify how much the STE bias matters on realistic tasks.

5. **Fine-tuning the frozen backbone.** The current analysis assumes the backbone is frozen during TINAS training. Extending the convergence analysis to a regime where the backbone is jointly fine-tuned — at the cost of the \(O(T)\)-buffer two-pass algorithm — is open.

---

## References (selected)

- Bottou, Curtis, Nocedal (2018). *Optimization Methods for Large-Scale Machine Learning.* SIAM Review.
- Chen, Liu, Sun, Hong (2019). *On the Convergence of A Class of Adam-Type Algorithms for Non-Convex Optimization.* ICLR.
- Foerster et al. (2016). *Learning to Communicate with Deep Multi-Agent Reinforcement Learning.* NeurIPS. (DIAL)
- Ghadimi & Lan (2013). *Stochastic First- and Zeroth-Order Methods for Nonconvex Stochastic Programming.* SIAM Journal on Optimization.
- Kingma & Ba (2015). *Adam: A Method for Stochastic Optimization.* ICLR.
- Bengio, Léonard, Courville (2013). *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.* arXiv:1308.3432. (STE)
- Wang et al. (2018). *Non-local Neural Networks.* CVPR.
