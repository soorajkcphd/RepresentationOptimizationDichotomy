import os
import sys
import time
import numpy as np
import scipy.linalg as la
from scipy.stats import ttest_1samp, pearsonr
from scipy.sparse.linalg import cg as scipy_cg
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

GLOBAL_SEED = 42
FIG_DIR = "figs_simods"
N_SEEDS = 5


def ensure_fig_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def save_fig(name):
    ensure_fig_dir()
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  [FIG] Saved {path}")
    plt.close()


class SO3:
    @staticmethod
    def hat(w):
        return np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0.0]
        ])

    @staticmethod
    def vee(X):
        return np.array([X[2, 1], X[0, 2], X[1, 0]])

    @staticmethod
    def exp(X):
        w = SO3.vee(X)
        theta = np.linalg.norm(w)
        if theta < 1e-10:
            return np.eye(3) + X
        K = X / theta
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    @staticmethod
    def project(M):
        return 0.5 * (M - M.T)


class SE3:
    @staticmethod
    def hat(xi):
        w, v = xi[:3], xi[3:]
        X = np.zeros((4, 4))
        X[:3, :3] = SO3.hat(w)
        X[:3, 3] = v
        return X

    @staticmethod
    def vee(X):
        return np.concatenate([SO3.vee(X[:3, :3]), X[:3, 3]])

    @staticmethod
    def exp(X):
        w = SO3.vee(X[:3, :3])
        v = X[:3, 3]
        theta = np.linalg.norm(w)
        T = np.eye(4)
        if theta < 1e-10:
            T[:3, :3] = np.eye(3) + X[:3, :3]
            T[:3, 3] = v
        else:
            K = X[:3, :3] / theta
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            V = np.eye(3) + ((1 - np.cos(theta)) / theta) * K + ((theta - np.sin(theta)) / theta) * (K @ K)
            T[:3, :3] = R
            T[:3, 3] = V @ v
        return T

    @staticmethod
    def log(T):
        R = T[:3, :3]
        t = T[:3, 3]
        trace_val = np.clip((np.trace(R) - 1) / 2, -1, 1)
        theta = np.arccos(trace_val)
        X = np.zeros((4, 4))
        if theta < 1e-6:
            X[:3, :3] = 0.5 * (R - R.T)
            X[:3, 3] = t
        else:
            K = theta / (2 * np.sin(theta)) * (R - R.T)
            X[:3, :3] = K
            K_n = K / theta
            V_inv = np.eye(3) - 0.5 * K + (1 / theta ** 2) * (1 - theta * np.cos(theta / 2) / (2 * np.sin(theta / 2))) * (K @ K)
            X[:3, 3] = V_inv @ t
        return X

    @staticmethod
    def project(M):
        X = np.zeros((4, 4))
        X[:3, :3] = SO3.project(M[:3, :3])
        X[:3, 3] = M[:3, 3]
        return X


class SO3JEnv:
    def __init__(self, J=10, horizon=40):
        self.J = J
        self.horizon = horizon
        self.gamma = 0.99
        self.reset()

    def reset(self):
        self.state = [SO3.exp(SO3.hat(np.random.randn(3) * 0.3)) for _ in range(self.J)]
        self.target = [SO3.exp(SO3.hat(np.random.randn(3) * 0.8)) for _ in range(self.J)]
        self.t = 0
        return self.get_obs()

    def get_obs(self):
        obs = []
        for s, tgt in zip(self.state, self.target):
            R_err = tgt.T @ s
            trace_val = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
            angle = np.arccos(trace_val)
            if angle < 1e-6:
                w = np.zeros(3)
            else:
                w = angle / (2 * np.sin(angle) + 1e-8) * SO3.vee(R_err - R_err.T)
            obs.extend(w)
        return np.array(obs)

    def step(self, action):
        action = np.clip(action, -2.0, 2.0)
        for j in range(self.J):
            w = action[3 * j:3 * (j + 1)] * 0.1
            self.state[j] = self.state[j] @ SO3.exp(SO3.hat(w))
            U, _, Vh = la.svd(self.state[j])
            self.state[j] = U @ Vh
        reward = 0.0
        for s, tgt in zip(self.state, self.target):
            R_err = tgt.T @ s
            trace_val = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
            reward -= np.arccos(trace_val) ** 2
        self.t += 1
        return self.get_obs(), reward, self.t >= self.horizon, {}

    @property
    def obs_dim(self):
        return 3 * self.J

    @property
    def act_dim(self):
        return 3 * self.J


class SE3Env:
    def __init__(self, horizon=40, B_theta=2.0, rot_scale=0.6, trans_scale=1.5):
        self.horizon = horizon
        self.gamma = 0.99
        self.B_theta = B_theta
        self.rot_scale = rot_scale
        self.trans_scale = trans_scale
        self.reset()

    def reset(self):
        xi_init = np.zeros(6)
        xi_init[:3] = np.random.randn(3) * 0.2
        xi_init[3:] = np.random.randn(3) * 0.3
        self.state = SE3.exp(SE3.hat(xi_init))
        xi_target = np.zeros(6)
        xi_target[:3] = np.random.randn(3) * self.rot_scale
        xi_target[3:] = np.random.randn(3) * self.trans_scale
        self.target = SE3.exp(SE3.hat(xi_target))
        self.t = 0
        return self.get_obs()

    def get_obs(self):
        T_err = np.linalg.solve(self.state, np.eye(4)) @ self.target
        return SE3.vee(SE3.log(T_err))

    def step(self, action):
        action = np.clip(action, -2.0, 2.0)
        xi = action * 0.1
        norm = np.linalg.norm(xi)
        if norm > self.B_theta:
            xi = self.B_theta * xi / norm
        self.state = self.state @ SE3.exp(SE3.hat(xi))
        T_err = np.linalg.solve(self.state, np.eye(4)) @ self.target
        log_err = SE3.log(T_err)
        reward = -np.linalg.norm(log_err, 'fro') ** 2
        self.t += 1
        return self.get_obs(), reward, self.t >= self.horizon, {}

    @property
    def obs_dim(self):
        return 6

    @property
    def act_dim(self):
        return 6


class LieLinearPolicy:
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.theta = np.zeros(act_dim)
        self.log_std = -0.5 * np.ones(act_dim)

    @property
    def n_params(self):
        return len(self.theta)

    def mean(self, s):
        return self.theta.copy()

    def sample(self, s):
        mu = self.mean(s)
        std = np.exp(self.log_std)
        a = mu + std * np.random.randn(self.act_dim)
        logp = -0.5 * np.sum(((a - mu) / std) ** 2 + 2 * np.log(std) + np.log(2 * np.pi))
        return a, logp

    def score(self, s, a):
        mu = self.mean(s)
        std = np.exp(self.log_std)
        return (a - mu) / (std ** 2)


class ControlledNoisePolicy:
    def __init__(self, obs_dim, act_dim, sigma_vec=None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.theta = np.zeros(act_dim)
        if sigma_vec is None:
            self.sigma = np.exp(-0.5) * np.ones(act_dim)
        else:
            self.sigma = np.array(sigma_vec, dtype=float)

    @property
    def n_params(self):
        return len(self.theta)

    def mean(self, s):
        return self.theta.copy()

    def sample(self, s):
        mu = self.mean(s)
        a = mu + self.sigma * np.random.randn(self.act_dim)
        logp = -0.5 * np.sum(((a - mu) / self.sigma) ** 2 + 2 * np.log(self.sigma) + np.log(2 * np.pi))
        return a, logp

    def score(self, s, a):
        mu = self.mean(s)
        return (a - mu) / (self.sigma ** 2)


class DiagFeaturePolicy:
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.theta = np.zeros(act_dim)
        self.log_std = -0.5 * np.ones(act_dim)

    @property
    def n_params(self):
        return len(self.theta)

    def mean(self, s):
        return self.theta * s[:self.act_dim]

    def sample(self, s):
        mu = self.mean(s)
        std = np.exp(self.log_std)
        a = mu + std * np.random.randn(self.act_dim)
        logp = -0.5 * np.sum(((a - mu) / std) ** 2 + 2 * np.log(std) + np.log(2 * np.pi))
        return a, logp

    def score(self, s, a):
        mu = self.mean(s)
        std = np.exp(self.log_std)
        return (a - mu) / (std ** 2) * s[:self.act_dim]


class AmbientLinearPolicy:
    def __init__(self, obs_dim, act_dim, k=15, anisotropic=True, rng_seed=999):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ambient_dim = act_dim * k
        self.theta = np.zeros(self.ambient_dim)
        rng = np.random.RandomState(rng_seed)
        P = rng.randn(self.act_dim, self.ambient_dim)
        P /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-8
        if anisotropic:
            P = P * np.linspace(0.1, 5.0, self.ambient_dim)
            P = np.diag(np.linspace(0.3, 3.0, self.act_dim)) @ P
        self.P = P
        self.log_std = -0.5 * np.ones(self.act_dim)

    @property
    def n_params(self):
        return len(self.theta)

    def mean(self, s):
        return self.P @ self.theta

    def sample(self, s):
        mu = self.mean(s)
        std = np.exp(self.log_std)
        a = mu + std * np.random.randn(self.act_dim)
        logp = -0.5 * np.sum(((a - mu) / std) ** 2 + 2 * np.log(std) + np.log(2 * np.pi))
        return a, logp

    def score(self, s, a):
        mu = self.mean(s)
        std = np.exp(self.log_std)
        return self.P.T @ ((a - mu) / (std ** 2))


def train_reinforce(policy, env, n_iters, lr, n_episodes=8, lr_decay=False):
    returns_history = []
    grad_norms = []
    for it in range(n_iters):
        current_lr = lr / np.sqrt(it + 1) if lr_decay else lr
        all_states, all_actions, all_returns = [], [], []
        ep_returns = []
        for _ in range(n_episodes):
            states, actions, rewards = [], [], []
            s = env.reset()
            done = False
            while not done:
                a, _ = policy.sample(s)
                s_next, r, done, _ = env.step(a)
                states.append(s)
                actions.append(a)
                rewards.append(r)
                s = s_next
            G = 0.0
            returns = []
            for r in reversed(rewards):
                G = r + env.gamma * G
                returns.insert(0, G)
            all_states.extend(states)
            all_actions.extend(actions)
            all_returns.extend(returns)
            ep_returns.append(sum(rewards))
        returns_history.append(np.mean(ep_returns))
        advantages = np.array(all_returns)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        grad = np.zeros_like(policy.theta)
        for s, a, adv in zip(all_states, all_actions, advantages):
            grad += adv * policy.score(s, a)
        grad /= len(all_states)
        grad_norms.append(np.linalg.norm(grad))
        policy.theta += current_lr * grad
    return returns_history, grad_norms


def estimate_fisher(policy, env, n_samples=500):
    d = policy.n_params
    F = np.zeros((d, d))
    for _ in range(n_samples):
        s = env.reset()
        a, _ = policy.sample(s)
        score = policy.score(s, a)
        F += np.outer(score, score)
    return F / n_samples


def compute_fisher_metrics(F):
    d = F.shape[0]
    eigvals = np.sort(np.real(la.eigvalsh(F)))[::-1]
    eigvals_pos = eigvals[eigvals > 1e-10]
    lam_max = eigvals_pos[0] if len(eigvals_pos) > 0 else 1.0
    lam_min = eigvals_pos[-1] if len(eigvals_pos) > 0 else 1e-10
    kappa = lam_max / lam_min
    lam_bar = np.trace(F) / d
    eps_F = la.norm(F - lam_bar * np.eye(d)) / (la.norm(F) + 1e-10)
    total = np.sum(eigvals_pos)
    r_eff = total ** 2 / (np.sum(eigvals_pos ** 2) + 1e-10)
    return {
        'kappa': kappa,
        'eps_F': eps_F,
        'r_eff': r_eff,
        'eigvals': eigvals_pos,
        'lam_bar': lam_bar
    }


def collect_fisher_and_alignment(policy, env, grad, n_fisher_samples=500):
    F = estimate_fisher(policy, env, n_samples=n_fisher_samples)
    metrics = compute_fisher_metrics(F)
    F_reg = F + 1e-4 * np.eye(policy.n_params)
    nat_grad = la.solve(F_reg, grad)
    align = np.dot(nat_grad, grad) / (la.norm(nat_grad) * la.norm(grad) + 1e-10)
    return metrics, align


def one_pg_iteration(policy, env, n_episodes=4):
    states, actions, returns_list = [], [], []
    ep_rets = []
    for _ in range(n_episodes):
        s = env.reset()
        done = False
        rewards = []
        while not done:
            a, _ = policy.sample(s)
            s_next, r, done, _ = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s_next
        G = 0.0
        for r in reversed(rewards):
            G = r + env.gamma * G
            returns_list.insert(0, G)
        ep_rets.append(sum(rewards))
    advs = np.array(returns_list)
    advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)
    grad = np.zeros_like(policy.theta)
    for s, a, adv in zip(states, actions, advs):
        grad += adv * policy.score(s, a)
    grad /= len(states)
    return grad, ep_rets


def exp1_fisher_alignment(J=10, n_iters=40, n_seeds=N_SEEDS):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Fisher-Metric Alignment (Table 4 + Fig 1)")
    print("=" * 70)

    all_align, all_kappa, all_eps, all_reff = [], [], [], []
    per_iter_eps = np.zeros((n_seeds, n_iters))
    per_iter_kappa = np.zeros((n_seeds, n_iters))
    per_iter_align = np.zeros((n_seeds, n_iters))

    for seed in range(n_seeds):
        np.random.seed(seed)
        print(f"  Seed {seed + 1}/{n_seeds}")
        env = SO3JEnv(J=J, horizon=30)
        policy = LieLinearPolicy(env.obs_dim, env.act_dim)
        for it in range(n_iters):
            grad, _ = one_pg_iteration(policy, env, n_episodes=4)
            metrics, align = collect_fisher_and_alignment(policy, env, grad)
            all_align.append(align)
            all_kappa.append(metrics['kappa'])
            all_eps.append(metrics['eps_F'])
            all_reff.append(metrics['r_eff'])
            per_iter_eps[seed, it] = metrics['eps_F']
            per_iter_kappa[seed, it] = metrics['kappa']
            per_iter_align[seed, it] = align
            policy.theta += 0.05 * grad
            if it % 10 == 0:
                print(f"    Iter {it:3d}: align={align:.3f} kappa={metrics['kappa']:.2f} "
                      f"eps_F={metrics['eps_F']:.3f} r_eff={metrics['r_eff']:.1f}")

    A = np.array(all_align)
    K = np.array(all_kappa)
    E = np.array(all_eps)
    R = np.array(all_reff)
    kappa_mean = np.mean(K)
    bound = 2 * np.sqrt(kappa_mean) / (kappa_mean + 1)
    ci = 1.96 * np.std(A) / np.sqrt(len(A))
    t_stat, p_val = ttest_1samp(A, 0.9)

    print("\n  TABLE 1 DATA:")
    print(f"    Mean alignment:  {np.mean(A):.3f}")
    print(f"    95% CI:          [{np.mean(A) - ci:.3f}, {np.mean(A) + ci:.3f}]")
    print(f"    Std. dev.:       {np.std(A):.3f}")
    print(f"    Range:           [{np.min(A):.3f}, {np.max(A):.3f}]")
    print(f"    kappa:           {np.mean(K):.2f} +/- {np.std(K):.2f}")
    print(f"    eps_F:           {np.mean(E):.2f} +/- {np.std(E):.2f}")
    print(f"    r_eff:           {np.mean(R):.1f} +/- {np.std(R):.1f} (of {3 * J})")
    print(f"    kappa-bound:     {bound:.3f}")
    print(f"    H0 align<=0.9:   t={t_stat:.2f}, p={p_val / 2:.2e}")
    print(f"    eps_F tracking:  {per_iter_eps[:, 0].mean():.2f} -> {per_iter_eps[:, -1].mean():.2f}")

    plt.figure(figsize=(5, 4))
    plt.hist(A, bins=20, color='tab:blue', alpha=0.8)
    plt.axvline(np.mean(A), color='black', linestyle='-', lw=2, label=f'mean={np.mean(A):.3f}')
    plt.axvline(bound, color='red', linestyle='--', lw=2, label=f'kappa-bound={bound:.3f}')
    plt.xlabel("cosine(natural gradient, vanilla gradient)")
    plt.ylabel("count")
    plt.title("Fisher-Metric Alignment")
    plt.legend()
    save_fig("exp1_fisher_alignment_hist.png")

    fig, axes = plt.subplots(3, 1, figsize=(5, 7), sharex=True)
    iters = np.arange(n_iters)
    plot_items = [
        (axes[0], per_iter_eps, 'eps_F', '(a) Isotropy deviation'),
        (axes[1], per_iter_kappa, 'kappa', '(b) Condition number'),
        (axes[2], per_iter_align, 'Alignment', '(c) Gradient alignment'),
    ]
    for ax, data, ylabel, title in plot_items:
        m = data.mean(axis=0)
        s = data.std(axis=0)
        ax.plot(iters, m, color='tab:blue')
        ax.fill_between(iters, m - s, m + s, alpha=0.2, color='tab:blue')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Iteration")
    fig.suptitle("Isotropy Tracking During Training", fontsize=12)
    save_fig("isotropy_tracking.png")

    plt.figure(figsize=(5, 4))
    plt.scatter(E, A, alpha=0.5, s=10)
    plt.xlabel('eps_F')
    plt.ylabel("Alignment")
    plt.title("Alignment vs. Fisher Isotropy Deviation")
    save_fig("exp1_alignment_vs_eps.png")

    return {'alignments': A, 'kappas': K, 'eps_F': E, 'r_eff': R}


def exp2_approximate_equivalence(d=20, n_points=100):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Approximate Equivalence (Section 6.2)")
    print("=" * 70)
    epsilons, alignments = [], []
    rng = np.random.RandomState(123)
    for _ in range(n_points):
        Q, _ = la.qr(rng.randn(d, d))
        lambdas = np.abs(1.0 + 0.5 * rng.randn() + rng.randn(d) * 0.15) + 0.2
        F = Q @ np.diag(lambdas) @ Q.T
        c_eff = np.trace(F) / d
        eps = la.norm(F - c_eff * np.eye(d)) / (la.norm(F) + 1e-10)
        g = rng.randn(d)
        nat_grad = la.solve(F, g)
        approx_grad = (1.0 / c_eff) * g
        align = np.dot(nat_grad, approx_grad) / (np.linalg.norm(nat_grad) * np.linalg.norm(approx_grad) + 1e-10)
        epsilons.append(eps)
        alignments.append(align)

    epsilons = np.array(epsilons)
    alignments = np.array(alignments)
    corr, p_val = pearsonr(epsilons, alignments)
    slope, intercept = np.polyfit(epsilons, alignments, 1)
    print(f"  Correlation: r = {corr:.3f}, p = {p_val:.2e}")
    print(f"  Fit: alignment ~ {intercept:.2f} + ({slope:.2f}) * eps_F")

    plt.figure(figsize=(5, 4))
    plt.scatter(epsilons, alignments, alpha=0.7, s=20, label="samples")
    eg = np.linspace(epsilons.min(), epsilons.max(), 100)
    plt.plot(eg, intercept + slope * eg, 'r-', lw=2, label="linear fit")
    plt.xlabel("eps_F")
    plt.ylabel("alignment")
    plt.title("Approximate Equivalence")
    plt.legend()
    save_fig("exp5_eps_vs_alignment.png")
    return {'corr': corr, 'slope': slope, 'intercept': intercept}


def exp3_anisotropy_ablation(J=10, n_iters=40, n_seeds=N_SEEDS):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Anisotropy Ablation (Table 5 tab:controlled_anisotropy + Fig 2 left)")
    print("=" * 70)

    d_g = 3 * J
    sigma_base = np.exp(-0.5)

    sigma_axis = sigma_base * np.ones(d_g)
    axis_factor = 1.5
    for j in range(J):
        sigma_axis[3 * j + 2] = sigma_base * np.sqrt(axis_factor)

    def make_sigma_correlated(kappa_M):
        scales = np.linspace(1.0, kappa_M, d_g)
        return sigma_base * np.sqrt(scales)

    conditions = [
        ('Uniform (baseline)', sigma_base * np.ones(d_g)),
        ('Axis-biased', sigma_axis),
        ('Correlated (kM=5)', make_sigma_correlated(5.0)),
        ('Correlated (kM=10)', make_sigma_correlated(10.0)),
    ]

    results = []
    for label, sigma_vec in conditions:
        print(f"\n  Condition: {label}")
        true_kappa = (np.max(1.0 / sigma_vec ** 2)) / (np.min(1.0 / sigma_vec ** 2))
        print(f"    True Fisher kappa = {true_kappa:.2f}")
        c_al, c_ka, c_ep, c_ret = [], [], [], []

        for seed in range(n_seeds):
            np.random.seed(seed)
            env = SO3JEnv(J=J, horizon=30)
            policy = ControlledNoisePolicy(env.obs_dim, env.act_dim, sigma_vec=sigma_vec)
            ep_rets_all = []

            for it in range(n_iters):
                grad, ep_rets = one_pg_iteration(policy, env, n_episodes=4)
                ep_rets_all.extend(ep_rets)
                metrics, align = collect_fisher_and_alignment(policy, env, grad)
                c_al.append(align)
                c_ka.append(metrics['kappa'])
                c_ep.append(metrics['eps_F'])
                policy.theta += 0.05 * grad
            c_ret.append(np.mean(ep_rets_all[-40:]))

        row = {
            'label': label,
            'kappa': np.mean(c_ka),
            'eps_F': np.mean(c_ep),
            'alignment': np.mean(c_al),
            'final_return': np.mean(c_ret)
        }
        results.append(row)
        print(f"    kappa={row['kappa']:.2f} eps_F={row['eps_F']:.2f} "
              f"align={row['alignment']:.3f} ret={row['final_return']:.1f}")

    print("\n  TABLE 2:")
    print(f"  {'Condition':<25} | {'kappa':>5} | {'eps_F':>5} | {'Align':>6} | {'Return':>8}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['label']:<25} | {r['kappa']:5.2f} | {r['eps_F']:5.2f} | "
              f"{r['alignment']:6.3f} | {r['final_return']:8.1f}")

    print("\n  Theory check:")
    for r in results:
        k = r['kappa']
        bound = 2 * np.sqrt(k) / (k + 1)
        status = 'OK' if r['alignment'] >= bound - 0.01 else 'VIOLATED'
        print(f"    {r['label']:<25}: align={r['alignment']:.3f} >= bound={bound:.3f} -> {status}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    kappas = [r['kappa'] for r in results]
    aligns = [r['alignment'] for r in results]
    eps_vals = [r['eps_F'] for r in results]
    rets = [r['final_return'] for r in results]

    ax1.plot(kappas, aligns, 'o-', color='tab:blue', ms=8)
    k_grid = np.linspace(min(kappas) * 0.9, max(kappas) * 1.1, 50)
    ax1.plot(k_grid, 2 * np.sqrt(k_grid) / (k_grid + 1), '--', color='red', label='2*sqrt(kappa)/(kappa+1)')
    ax1.set_xlabel('kappa')
    ax1.set_ylabel('Alignment')
    ax1.set_title('(a) Alignment vs. kappa')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(eps_vals, rets, 's-', color='tab:red', ms=8)
    ax2.set_xlabel('eps_F')
    ax2.set_ylabel('Final Return')
    ax2.set_title('(b) Return vs. eps_F')
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Controlled Anisotropy Ablation", fontsize=12)
    save_fig("controlled_anisotropy.png")
    return results


def exp4_sample_efficiency(J=10, n_iters=400, n_seeds=N_SEEDS):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 4: Sample Efficiency (Fig 2 right) | Lie:{3 * J}p vs Amb:{45 * J}p")
    print("=" * 70)
    lie_curves, amb_curves = [], []
    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}")
        np.random.seed(seed)
        env = SO3JEnv(J=J, horizon=30)
        pol = LieLinearPolicy(env.obs_dim, env.act_dim)
        ret, _ = train_reinforce(pol, env, n_iters=n_iters, lr=0.25, n_episodes=8)
        lie_curves.append(ret)
        print(f"    Lie  final(last40): {np.mean(ret[-40:]):.2f}")

        np.random.seed(seed + 1000)
        env = SO3JEnv(J=J, horizon=30)
        pol = AmbientLinearPolicy(env.obs_dim, env.act_dim, k=15, anisotropic=True)
        ret, _ = train_reinforce(pol, env, n_iters=n_iters, lr=0.25, n_episodes=8)
        amb_curves.append(ret)
        print(f"    Amb  final(last40): {np.mean(ret[-40:]):.2f}")

    lie_curves = np.array(lie_curves)
    amb_curves = np.array(amb_curves)
    lie_final = np.mean(lie_curves[:, -40:])
    amb_final = np.mean(amb_curves[:, -40:])
    lie_auc = np.mean(np.sum(lie_curves, axis=1))
    amb_auc = np.mean(np.sum(amb_curves, axis=1))
    print(f"\n  Lie final: {lie_final:.2f}, Amb final: {amb_final:.2f}")
    print(f"  AUC ratio (Lie/Amb): {lie_auc / (amb_auc + 1e-10):.2f}")

    T = lie_curves.shape[1]
    iters = np.arange(T)
    plt.figure(figsize=(5, 4))
    for curves, label, c in [
        (lie_curves, "Lie policy", "tab:blue"),
        (amb_curves, "Ambient policy", "tab:orange")
    ]:
        m = curves.mean(axis=0)
        s = curves.std(axis=0)
        plt.plot(iters, m, label=label, color=c)
        plt.fill_between(iters, m - s, m + s, alpha=0.2, color=c)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Episodic Return")
    plt.title("Sample Efficiency: Lie vs Ambient")
    plt.legend()
    save_fig("exp3_learning_curves.png")

    fig, ax1 = plt.subplots()
    x = np.arange(2)
    w = 0.35
    ax1.bar(x - w / 2, [lie_final, amb_final], w, label="final return", color="tab:blue")
    ax1.set_ylabel("final return")
    ax2 = ax1.twinx()
    ax2.bar(x + w / 2, [lie_auc, amb_auc], w, label="AUC", color="tab:orange")
    ax2.set_ylabel("AUC")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Lie", "Ambient"])
    fig.suptitle("Final Performance and AUC")
    ax1.legend(loc="upper left")
    save_fig("exp3_bar_stats.png")
    return {'lie_curves': lie_curves, 'amb_curves': amb_curves}


def exp5_convergence_rate(d=30, T_values=None, n_seeds=N_SEEDS):
    if T_values is None:
        T_values = [200, 400, 800, 1600]
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Convergence Rate deterministic (Fig 3 left)")
    print("=" * 70)
    results = {T: [] for T in T_values}
    for T in T_values:
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            Q, _ = la.qr(rng.randn(d, d))
            A = Q @ np.diag(np.linspace(1.0, 5.0, d)) @ Q.T
            theta = rng.randn(d)
            sigma = 0.5
            c = 0.8
            grad_sq = []
            for t in range(T):
                gt = A @ theta
                theta -= (c / np.sqrt(t + 1)) * (gt + sigma * rng.randn(d))
                grad_sq.append(np.linalg.norm(gt) ** 2)
            results[T].append(np.mean(grad_sq))

    T_arr = np.array(T_values, dtype=float)
    means = np.array([np.mean(results[T]) for T in T_values])
    print(f"\n  {'T':>6} | {'E[||grad||^2]':>14} | {'1/sqrt(T)':>10}")
    print("  " + "-" * 38)
    for T, m in zip(T_values, means):
        print(f"  {T:6d} | {m:14.6f} | {1 / np.sqrt(T):10.4f}")
    log_T = np.log(T_arr)
    log_m = np.log(means + 1e-15)
    slope, _ = np.polyfit(log_T, log_m, 1)
    print(f"\n  Fitted slope: {slope:.3f} (theory: -0.5)")

    plt.figure(figsize=(5, 4))
    plt.plot(log_T, log_m, 'o-', color='tab:blue', ms=8)
    plt.plot(log_T, log_m[0] - 0.5 * (log_T - log_T[0]), '--', color='gray', label='O(T^{-1/2})')
    plt.xlabel("log T")
    plt.ylabel("log E[||grad J||^2]")
    plt.title(f"Convergence Rate (slope = {slope:.2f})")
    plt.legend()
    save_fig("exp4_convergence_loglog.png")
    return {'slope': slope}


def exp5b_stochastic_convergence_rate(d=30, T_values=None, n_seeds=N_SEEDS, eta=0.25, sigma_g=1.0):
    if T_values is None:
        T_values = [200, 400, 800, 1600]

    print("\n" + "=" * 70)
    print("EXPERIMENT 5b: Stochastic Quadratic Proxy Convergence Rate (Fig 3 right)")
    print(f"  d_g={d}, eta={eta}, sigma_g={sigma_g}, seeds={n_seeds}")
    print("=" * 70)

    results = {T: [] for T in T_values}

    for T in T_values:
        lr = eta / np.sqrt(T)
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            Q, _ = la.qr(rng.randn(d, d))
            A = Q @ np.diag(np.linspace(1.0, 5.0, d)) @ Q.T
            theta = rng.randn(d) * 0.1
            grad_sq_traj = []
            for t in range(T):
                g_true = A @ theta
                g_noisy = g_true + sigma_g * rng.randn(d)
                grad_sq_traj.append(float(np.linalg.norm(g_true) ** 2))
                theta -= lr * g_noisy
            results[T].append(float(np.mean(grad_sq_traj)))

        mean_T = float(np.mean(results[T]))
        std_T = float(np.std(results[T]))
        print(f"  T={T:5d}, lr={lr:.5f}  ->  mean E[||grad||^2] = {mean_T:.4f} +/- {std_T:.4f}")

    T_arr = np.array(T_values, dtype=float)
    means = np.array([float(np.mean(results[T])) for T in T_values])
    log_T = np.log(T_arr)
    log_m = np.log(means + 1e-15)
    slope, intercept = np.polyfit(log_T, log_m, 1)
    print(f"\n  Fitted slope: {slope:.3f}  (theory: -0.50)")
    print(f"\n  {'T':>6} | {'mean E[||grad||^2]':>18} | {'1/sqrt(T)':>10}")
    print("  " + "-" * 42)
    for T, m in zip(T_values, means):
        print(f"  {T:6d} | {m:18.6f} | {1.0 / np.sqrt(T):10.4f}")

    plt.figure(figsize=(5, 4))
    plt.plot(log_T, log_m, 'o-', color='tab:blue', ms=8)
    plt.plot(log_T, log_m[0] - 0.5 * (log_T - log_T[0]), '--', color='gray', label='O(T^{-1/2})')
    plt.xlabel("log T")
    plt.ylabel("log E[||grad J||^2]")
    plt.title(f"Convergence Rate (slope = {slope:.2f})")
    plt.legend()
    save_fig("exp4_convergence_loglog_stochastic.png")

    return {'slope': slope, 'means': means.tolist(), 'T_values': T_values}


def exp6_timing(J_values=None):
    if J_values is None:
        J_values = [5, 10, 30]  # matches Table 5 (tab:timing) in paper
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Computational Efficiency (Table 5 tab:timing)")
    print("=" * 70)
    print(f"\n  {'J':>4} | {'d_g':>5} | {'Fisher(us)':>10} | {'Proj(us)':>9} | {'Speedup':>8}")
    print("  " + "-" * 50)
    results = {}
    for J in J_values:
        d_g = 3 * J
        tf = []
        for _ in range(200):
            A = np.random.randn(d_g, d_g)
            F = A @ A.T + 0.1 * np.eye(d_g)
            g = np.random.randn(d_g)
            t0 = time.perf_counter()
            la.solve(F, g)
            tf.append(time.perf_counter() - t0)
        tp = []
        for _ in range(200):
            M = np.random.randn(3 * J, 3 * J)
            t0 = time.perf_counter()
            for j in range(J):
                blk = M[3 * j:3 * (j + 1), 3 * j:3 * (j + 1)]
                M[3 * j:3 * (j + 1), 3 * j:3 * (j + 1)] = SO3.project(blk)
            tp.append(time.perf_counter() - t0)
        fu = np.mean(tf) * 1e6
        pu = np.mean(tp) * 1e6
        sp = fu / pu
        results[J] = {'fisher': fu, 'proj': pu, 'speedup': sp}
        print(f"  {J:4d} | {d_g:5d} | {fu:10.1f} | {pu:9.1f} | {sp:7.1f}x")

    J_arr = np.array(sorted(results.keys()))
    plt.figure(figsize=(5, 4))
    plt.plot(J_arr, [results[J]['fisher'] for J in J_arr], 'o-', label="Fisher inversion")
    plt.plot(J_arr, [results[J]['proj'] for J in J_arr], 's-', label="Lie projection")
    plt.xlabel("J")
    plt.ylabel("Time (us)")
    plt.title("Timing Comparison")
    plt.legend()
    save_fig("exp2_timing.png")

    plt.figure(figsize=(5, 4))
    plt.plot(J_arr, [results[J]['speedup'] for J in J_arr], 'o-')
    plt.xlabel("J")
    plt.ylabel("Speedup (Fisher/Proj)")
    plt.title("Computational Speedup")
    save_fig("exp2_speedup.png")
    return results


def exp7_scalability(J_values=None, n_iters=200, n_seeds=N_SEEDS):
    if J_values is None:
        J_values = [5, 10, 15, 20, 30]
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Scalability Ablation (Table 6 tab:scalability)")
    print("=" * 70)
    results = []
    for J in J_values:
        d_g = 3 * J
        n_fisher = max(500, 5 * d_g)
        print(f"\n  J={J}, d_g={d_g}, Fisher samples={n_fisher}")
        all_al, all_ep, lie_aucs, amb_aucs = [], [], [], []

        for seed in range(n_seeds):
            np.random.seed(seed)
            env = SO3JEnv(J=J, horizon=30)
            policy = LieLinearPolicy(env.obs_dim, env.act_dim)
            for it in range(40):
                grad, _ = one_pg_iteration(policy, env, n_episodes=4)
                metrics, align = collect_fisher_and_alignment(policy, env, grad, n_fisher_samples=n_fisher)
                all_al.append(align)
                all_ep.append(metrics['eps_F'])
                policy.theta += 0.05 * grad

            np.random.seed(seed)
            env = SO3JEnv(J=J, horizon=30)
            pol = LieLinearPolicy(env.obs_dim, env.act_dim)
            ret, _ = train_reinforce(pol, env, n_iters=n_iters, lr=0.25, n_episodes=8)
            lie_aucs.append(np.sum(ret))

            np.random.seed(seed + 1000)
            env = SO3JEnv(J=J, horizon=30)
            pol = AmbientLinearPolicy(env.obs_dim, env.act_dim, k=15, anisotropic=True)
            ret, _ = train_reinforce(pol, env, n_iters=n_iters, lr=0.25, n_episodes=8)
            amb_aucs.append(np.sum(ret))

        auc_ratio = np.mean(lie_aucs) / (np.mean(amb_aucs) + 1e-10)

        tf, tp = [], []
        for _ in range(200):
            A = np.random.randn(d_g, d_g)
            F = A @ A.T + 0.1 * np.eye(d_g)
            g = np.random.randn(d_g)
            t0 = time.perf_counter()
            la.solve(F, g)
            tf.append(time.perf_counter() - t0)
        for _ in range(200):
            M = np.random.randn(3 * J, 3 * J)
            t0 = time.perf_counter()
            for j in range(J):
                b = M[3 * j:3 * (j + 1), 3 * j:3 * (j + 1)]
                M[3 * j:3 * (j + 1), 3 * j:3 * (j + 1)] = SO3.project(b)
            tp.append(time.perf_counter() - t0)
        speedup = np.mean(tf) / (np.mean(tp) + 1e-15)

        row = {
            'J': J,
            'd_g': d_g,
            'align_mean': np.mean(all_al),
            'align_std': np.std(all_al),
            'auc_ratio': auc_ratio,
            'speedup': speedup,
            'eps_F': np.mean(all_ep)
        }
        results.append(row)
        print(f"    align={row['align_mean']:.3f}+/-{row['align_std']:.3f} "
              f"AUC={row['auc_ratio']:.2f} sp={row['speedup']:.1f}x eF={row['eps_F']:.2f}")

    print("\n  TABLE 5:")
    print(f"  {'J':>3} | {'d_g':>4} | {'Alignment':>15} | {'AUC':>5} | {'Speedup':>8} | {'eF':>5}")
    print("  " + "-" * 55)
    for r in results:
        print(f"  {r['J']:3d} | {r['d_g']:4d} | {r['align_mean']:.3f} +/- {r['align_std']:.3f} | "
              f"{r['auc_ratio']:5.2f} | {r['speedup']:7.1f}x | {r['eps_F']:5.2f}")
    return results


def exp8_se3(n_iters=40, n_seeds=N_SEEDS):
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: SE(3) Non-Compact (Table 7 tab:se3_results)")
    print("=" * 70)
    configs = [('so(3) (compact)', 'so3'), ('se(3) (non-compact)', 'se3')]
    results = []
    for label, algebra in configs:
        print(f"\n  Algebra: {label}")
        c_al, c_ka, c_ep = [], [], []
        for seed in range(n_seeds):
            np.random.seed(seed)
            if algebra == 'so3':
                env = SO3JEnv(J=1, horizon=30)
            else:
                env = SE3Env(horizon=30, B_theta=2.0, rot_scale=0.7, trans_scale=1.0)

            policy = DiagFeaturePolicy(env.obs_dim, env.act_dim)

            for it in range(n_iters):
                grad, _ = one_pg_iteration(policy, env, n_episodes=4)
                metrics, align = collect_fisher_and_alignment(policy, env, grad)
                c_al.append(align)
                c_ka.append(metrics['kappa'])
                c_ep.append(metrics['eps_F'])
                policy.theta += 0.05 * grad

        row = {
            'label': label,
            'd_g': env.act_dim,
            'align_mean': np.mean(c_al),
            'align_std': np.std(c_al),
            'kappa_mean': np.mean(c_ka),
            'kappa_std': np.std(c_ka),
            'eps_F': np.mean(c_ep)
        }
        results.append(row)
        print(f"    align={row['align_mean']:.3f}+/-{row['align_std']:.3f} "
              f"kappa={row['kappa_mean']:.2f}+/-{row['kappa_std']:.2f} eps_F={row['eps_F']:.2f}")

    print("\n  Divergence test (SE(3) with/without radius projection)...")
    N_DIV_SEEDS = 5
    n_div_iters = 150
    lr_div = 3.0
    B_theta_param = 2.0

    param_norms_div, grad_norms_div = [], []
    param_norms_stab, grad_norms_stab = [], []

    for seed in range(N_DIV_SEEDS):
        np.random.seed(200 + seed)
        env_no = SE3Env(horizon=30, B_theta=1e6, rot_scale=0.7, trans_scale=1.0)
        pol_no = DiagFeaturePolicy(env_no.obs_dim, env_no.act_dim)
        pn, gn = [], []
        for it in range(n_div_iters):
            grad, _ = one_pg_iteration(pol_no, env_no, n_episodes=4)
            pn.append(np.linalg.norm(pol_no.theta))
            gn.append(np.linalg.norm(grad))
            pol_no.theta += lr_div * grad
        param_norms_div.append(pn)
        grad_norms_div.append(gn)

        np.random.seed(200 + seed)
        env_st = SE3Env(horizon=30, B_theta=2.0, rot_scale=0.7, trans_scale=1.0)
        pol_st = DiagFeaturePolicy(env_st.obs_dim, env_st.act_dim)
        pn, gn = [], []
        for it in range(n_div_iters):
            grad, _ = one_pg_iteration(pol_st, env_st, n_episodes=4)
            pn.append(np.linalg.norm(pol_st.theta))
            gn.append(np.linalg.norm(grad))
            pol_st.theta += lr_div * grad
            tnorm = np.linalg.norm(pol_st.theta)
            if tnorm > B_theta_param:
                pol_st.theta *= B_theta_param / tnorm
        param_norms_stab.append(pn)
        grad_norms_stab.append(gn)

    pn_div = np.array(param_norms_div)
    gn_div = np.array(grad_norms_div)
    pn_stab = np.array(param_norms_stab)
    gn_stab = np.array(grad_norms_stab)
    iters_div = np.arange(n_div_iters)

    print(f"    No proj  -- final ||theta||: {pn_div[:, -1].mean():.2f}  "
          f"final ||grad||: {gn_div[:, -1].mean():.2f}")
    print(f"    B_theta=2 -- final ||theta||: {pn_stab[:, -1].mean():.2f}  "
          f"final ||grad||: {gn_stab[:, -1].mean():.2f}")
    degraded = gn_div[:, -1].mean() > 2 * gn_stab[:, -1].mean()
    status_str = 'grows significantly' if degraded else 'comparable'
    print(f"    Gradient norm {status_str} without projection "
          f"({gn_div[:, -1].mean():.2f} vs {gn_stab[:, -1].mean():.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    m = pn_div.mean(axis=0)
    s = pn_div.std(axis=0)
    ax.plot(iters_div, m, color='tab:red', lw=2, label='No projection (B_theta=inf)')
    ax.fill_between(iters_div, np.maximum(m - s, 0), m + s, alpha=0.2, color='tab:red')
    m2 = pn_stab.mean(axis=0)
    s2 = pn_stab.std(axis=0)
    ax.plot(iters_div, m2, color='tab:blue', lw=2, label='With projection (B_theta=2)')
    ax.fill_between(iters_div, np.maximum(m2 - s2, 0), m2 + s2, alpha=0.2, color='tab:blue')
    ax.axhline(B_theta_param, color='tab:blue', linestyle=':', lw=1.5, alpha=0.8,
               label=f'B_theta = {B_theta_param} bound')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("||theta||_F")
    ax.set_title("(a) Parameter norm over training")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    R_div = pn_div.mean(axis=0)
    R_stab = pn_stab.mean(axis=0)
    # se(3) has no hyperbolic elements: L(R) = O(R^2) polynomial, not exp(2R)
    L_div = R_div ** 2
    L_stab = R_stab ** 2
    L_div = L_div / (L_div[0] + 1e-10)
    L_stab = L_stab / (L_stab[0] + 1e-10)
    ax.semilogy(iters_div, L_div, color='tab:red', lw=2, label='No projection: R^2 grows (polynomial)')
    ax.semilogy(iters_div, L_stab, color='tab:blue', lw=2, label='With projection: R^2 bounded')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative L(R) ~ R^2 [log scale]")
    ax.set_title("(b) Lipschitz constant L(R) = O(R^2) for se(3)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    fig.suptitle("SE(3): Without radius projection, L(R)=O(R^2) grows polynomially", fontsize=11)
    save_fig("exp_se3_divergence.png")

    print("\n  TABLE 8:")
    print(f"  {'Algebra':<22} | {'d_g':>4} | {'Alignment':>15} | {'kappa':>12} | {'eps_F':>5}")
    print("  " + "-" * 68)
    for r in results:
        print(f"  {r['label']:<22} | {r['d_g']:4d} | {r['align_mean']:.3f} +/- {r['align_std']:.3f} | "
              f"{r['kappa_mean']:5.2f} +/- {r['kappa_std']:5.2f} | {r['eps_F']:5.2f}")

    print("\n  Theory check:")
    for r in results:
        k = r['kappa_mean']
        bound = 2 * np.sqrt(k) / (k + 1)
        status = 'OK' if r['align_mean'] >= bound - 0.01 else 'CHECK'
        print(f"    {r['label']}: align={r['align_mean']:.3f} >= bound={bound:.3f} -> {status}")

    return results


def exp9_method_comparison(J=10, n_iters=200, n_seeds=N_SEEDS):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 9: Method Comparison (LPG:{3 * J}p | Amb:{9 * J}p | NatGrad:{3 * J}p+CG)")
    print("=" * 70)
    lpg_c, amb_c, nat_c = [], [], []
    lpg_t, amb_t, nat_t = [], [], []

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}")
        for method in ['lpg', 'ambient', 'natgrad']:
            np.random.seed(seed)
            env = SO3JEnv(J=J, horizon=30)
            if method == 'ambient':
                policy = AmbientLinearPolicy(env.obs_dim, env.act_dim, k=3, anisotropic=False)
            else:
                policy = LieLinearPolicy(env.obs_dim, env.act_dim)
            ret = []
            stimes = []
            for it in range(n_iters):
                grad, ep_rets = one_pg_iteration(policy, env, n_episodes=8)
                ret.append(np.mean(ep_rets))
                t0 = time.perf_counter()
                if method == 'lpg':
                    for j in range(J):
                        blk = grad[3 * j:3 * (j + 1)]
                        M = SO3.hat(blk)
                        M = SO3.project(M)
                        grad[3 * j:3 * (j + 1)] = SO3.vee(M)
                    policy.theta += 0.25 * grad
                elif method == 'ambient':
                    policy.theta += 0.25 * grad
                else:
                    F = estimate_fisher(policy, env, n_samples=200)
                    F_reg = F + 1e-4 * np.eye(policy.n_params)
                    ng, _ = scipy_cg(F_reg, grad, maxiter=10)
                    policy.theta += 0.25 * ng
                stimes.append(time.perf_counter() - t0)

            mean_t = np.mean(stimes) * 1e6
            if method == 'lpg':
                lpg_c.append(ret)
                lpg_t.append(mean_t)
                print(f"    LPG:     final={np.mean(ret[-20:]):.1f}  step={mean_t:.1f}us")
            elif method == 'ambient':
                amb_c.append(ret)
                amb_t.append(mean_t)
                print(f"    Ambient: final={np.mean(ret[-20:]):.1f}  step={mean_t:.1f}us")
            else:
                nat_c.append(ret)
                nat_t.append(mean_t)
                print(f"    NatGrad: final={np.mean(ret[-20:]):.1f}  step={mean_t:.1f}us")

    lpg_c = np.array(lpg_c)
    amb_c = np.array(amb_c)
    nat_c = np.array(nat_c)

    print("\n  RESULTS TABLE:")
    print(f"  {'Method':<16} | {'Params':>6} | {'Final Return':>14} | {'Step(us)':>10}")
    print("  " + "-" * 55)
    for nm, cu, ti, np_ in [
        ("LPG (ours)", lpg_c, lpg_t, 3 * J),
        ("Ambient PG (3x)", amb_c, amb_t, 9 * J),
        ("Natural Grad", nat_c, nat_t, 3 * J),
    ]:
        fr = np.mean(cu[:, -20:])
        frs = np.std(np.mean(cu[:, -20:], axis=1))
        print(f"  {nm:<16} | {np_:>6} | {fr:>7.1f} +/- {frs:>4.1f} | {np.mean(ti):>10.1f}")

    T = lpg_c.shape[1]
    iters = np.arange(T)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nm, cu, col, ls in [
        ("LPG (ours)", lpg_c, "tab:blue", "-"),
        ("Ambient PG (3x)", amb_c, "tab:orange", "--"),
        ("Natural Grad (CG)", nat_c, "tab:green", "-."),
    ]:
        m = cu.mean(axis=0)
        s = cu.std(axis=0)
        ax.plot(iters, m, label=nm, color=col, linestyle=ls, lw=2)
        ax.fill_between(iters, m - s, m + s, alpha=0.15, color=col)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Episodic Return")
    ax.set_title(f"Method Comparison on SO(3)^{J}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig("exp9_method_comparison.png")

    lpg_us = np.mean(lpg_t)
    amb_us = np.mean(amb_t)
    nat_us = np.mean(nat_t)

    T = lpg_c.shape[1]
    lpg_wall = np.arange(T) * lpg_us * 1e-6
    amb_wall = np.arange(T) * amb_us * 1e-6
    nat_wall = np.arange(T) * nat_us * 1e-6

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nm, cu, wall, col, ls in [
        ("LPG (ours)", lpg_c, lpg_wall, "tab:blue", "-"),
        ("Ambient PG (3x)", amb_c, amb_wall, "tab:orange", "--"),
        ("Natural Grad (CG)", nat_c, nat_wall, "tab:green", "-."),
    ]:
        m = cu.mean(axis=0)
        s = cu.std(axis=0)
        ax.plot(wall, m, label=nm, color=col, linestyle=ls, lw=2)
        ax.fill_between(wall, m - s, m + s, alpha=0.15, color=col)

    ax.set_xscale("log")
    ax.set_xlabel("Cumulative wall-clock time (s, log scale)")
    ax.set_ylabel("Mean Episodic Return")
    ax.set_title(f"Return vs. Wall-Clock Time on SO(3)^{J}")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    nat_final = np.mean(nat_c[:, -20:])
    lpg_mean = lpg_c.mean(axis=0)
    crossing = np.searchsorted(lpg_mean[::-1] < nat_final, True)
    if 0 < crossing < T:
        cx = lpg_wall[T - 1 - crossing]
        ax.axvline(cx, color='tab:blue', linestyle=':', alpha=0.7)
        ax.text(cx * 1.15, nat_final * 0.97,
                f"LPG reaches NG level\n@ {cx:.2f} s",
                fontsize=8, color='tab:blue')

    save_fig("exp9_wallclock_comparison.png")
    return {'lpg': lpg_c, 'amb': amb_c, 'nat': nat_c}


ALL_EXPERIMENTS = {
    1: ('Fisher alignment (Table 4 + Fig 1)', exp1_fisher_alignment),
    2: ('Approximate equivalence (Sec 9.2)', exp2_approximate_equivalence),
    3: ('Anisotropy ablation (Table 5 + Fig 2 left)', exp3_anisotropy_ablation),
    4: ('Sample efficiency (Fig 2 right)', exp4_sample_efficiency),
    5: ('Convergence rate deterministic (Fig 3 left)', exp5_convergence_rate),
    6: ('Computational efficiency (Table 5 tab:timing)', exp6_timing),
    7: ('Scalability ablation (Table 6 tab:scalability)', exp7_scalability),
    8: ('SE(3) non-compact (Table 7 tab:se3_results)', exp8_se3),
    9: ('Method comparison (Table 8 + Fig 5)', exp9_method_comparison),
    10: ('Convergence rate stochastic quadratic proxy (Fig 3 right)', exp5b_stochastic_convergence_rate),
}


def run_all(selected=None):
    np.random.seed(GLOBAL_SEED)
    print("\n" + "=" * 80)
    print("UNIFIED EXPERIMENTAL VALIDATION - SIMODS PAPER")
    to_run = selected if selected else sorted(ALL_EXPERIMENTS.keys())
    print(f"Running experiments: {to_run}")
    print("=" * 80)
    results = {}
    for eid in to_run:
        if eid not in ALL_EXPERIMENTS:
            print(f"\n  WARNING: Experiment {eid} not found")
            continue
        name, func = ALL_EXPERIMENTS[eid]
        print(f"\n{'#' * 80}\n# EXPERIMENT {eid}: {name}\n{'#' * 80}")
        results[eid] = func()
    print(f"\n{'=' * 80}\nALL COMPLETE. Figures saved in {FIG_DIR}/\n{'=' * 80}")
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_all([int(x) for x in sys.argv[1:]])
    else:
        run_all()
