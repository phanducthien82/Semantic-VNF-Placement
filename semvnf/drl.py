"""Learning-based baselines (Section 4.1, Table 9): DDPG and GCN-PPO.

Both agents place the VNFs of a request sequentially (one decision per VNF) and route every
virtual link on the latency-shortest path with enough residual bandwidth.  They receive the
same semantic domains C_i as the other methods, as an action mask (so no policy violation
can be proposed), and the same objective: the reward of a step is minus the increment of J
(Eq. (24)) caused by the step, and a request that cannot be completed (no path, capacity or
SLA violation) ends with a penalty of -2 (about twice the typical |J| of a feasible request;
the -1000 of the first submission was rescaled to the reward range).

DDPG   actor/critic MLPs (256-128-64, ReLU); the actor outputs one logit per server; the executed
       action is the masked argmax of logits + Gaussian exploration noise; the critic takes the
       one-hot action and the actor is trained through the masked softmax relaxation.
GCN-PPO two GCN layers (64 units) on the substrate graph; masked categorical policy over
       servers; value head on mean-pooled embeddings; PPO-clip (0.2), entropy 0.01, GAE.

Timing (reply to R2 comment 12): training time is measured once per agent; at decision time
t_pre (state construction), t_inf (forward pass) and routing are measured separately; model
loading time is measured when the evaluation process loads the checkpoint.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from .model import Instance

torch.set_num_threads(1)
FAIL_PENALTY = 2.0
N_FEAT = 10
N_GLOBAL = 9


# ---------------------------------------------------------------------------------------- state
class Episode:
    """Sequential placement of one request; exposes features, mask and step rewards."""

    def __init__(self, inst: Instance, router):
        self.inst, self.router = inst, router
        r = inst.req
        self.n, self.N = inst.n, inst.N
        self.res_cpu, self.res_mem, self.res_bw = inst.res_cpu.copy(), inst.res_mem.copy(), inst.res_bw.copy()
        self.used = np.zeros(self.N, dtype=bool)
        self.prev = r.ingress
        self.i = 0
        self.lat = 0.0
        self.servers, self.paths = [], []
        self.dist = router.dist()
        self.cpu_cap = inst.sub.cpu_cap
        self.mem_cap = inst.sub.mem_cap
        self.L = r.sla_ms
        self.done = False
        self.failed = False

    def mask(self) -> np.ndarray:
        i, inst = self.i, self.inst
        m = inst.cand[i] & (self.res_cpu >= inst.cpu[i] - 1e-9) & (self.res_mem >= inst.mem[i] - 1e-9)
        return m & np.isfinite(self.dist[self.prev])

    def features(self) -> tuple[np.ndarray, np.ndarray]:
        inst, i = self.inst, self.i
        d_prev = np.minimum(self.dist[self.prev] / self.L, 5.0)
        d_egr = np.minimum(self.dist[:, inst.req.egress] / self.L, 5.0)
        util = 1.0 - self.res_cpu / self.cpu_cap
        f = np.stack([self.res_cpu / self.cpu_cap, self.res_mem / self.mem_cap,
                      (inst.active | self.used).astype(float), self.mask().astype(float),
                      np.nan_to_num(d_prev, posinf=5.0), np.nan_to_num(d_egr, posinf=5.0),
                      np.minimum(inst.c_x[i], 5.0), np.minimum(inst.c_y, 5.0) * (~self.used),
                      util, inst.d_proc[i] / self.L], axis=1).astype(np.float32)
        tpl = ["Web", "URLLC", "mIoT", "Enterprise", "eMBB"]
        g = np.zeros(N_GLOBAL, dtype=np.float32)
        g[0] = inst.cpu[i] / 8.0
        g[1] = (self.n - i) / 5.0
        g[2] = max(0.0, 1.0 - self.lat / self.L)
        g[3] = inst.req.bw / 600.0
        g[4 + tpl.index(inst.req.template)] = 1.0
        return f, g

    def step(self, j: int) -> float:
        """Place VNF i on server j; returns the step reward."""
        inst, i, bw = self.inst, self.i, self.inst.req.bw
        cost = float(inst.c_x[i, j]) + (float(inst.c_y[j]) if not self.used[j] else 0.0)
        if inst.c_h is not None:                       # headroom term of J, incremental per step
            cap = self.cpu_cap[j]
            load0 = inst.used_cpu[j] + (inst.res_cpu[j] - self.res_cpu[j])
            ex_b = max(0.0, load0 / cap - inst.theta_h) - inst.excess0[j]
            ex_a = max(0.0, (load0 + inst.cpu[i]) / cap - inst.theta_h) - inst.excess0[j]
            cost += float(inst.c_h_vec[j]) * (max(0.0, ex_a) - max(0.0, ex_b))
        p, plat = self.router.shortest(self.prev, j, bw, self.res_bw)
        if p is None:
            self.done = self.failed = True
            return -FAIL_PENALTY
        for a in p:
            self.res_bw[a] -= bw
        cost += float(inst.c_f[p].sum()) if p else 0.0
        self.lat += plat + inst.d_proc[i, j]
        self.res_cpu[j] -= inst.cpu[i]
        self.res_mem[j] -= inst.mem[i]
        self.used[j] = True
        self.servers.append(int(j))
        self.paths.append(p)
        self.prev = int(j)
        self.i += 1
        if self.i == self.n:
            q, qlat = self.router.shortest(self.prev, inst.req.egress, bw, self.res_bw)
            self.done = True
            if q is None:
                self.failed = True
                return -FAIL_PENALTY
            self.paths.append(q)
            self.lat += qlat
            cost += float(inst.c_f[q].sum()) if q else 0.0
            if self.lat > inst.req.sla_ms + 1e-9:
                self.failed = True
                return -FAIL_PENALTY
        return -cost


# ---------------------------------------------------------------------------------------- networks
def mlp(inp, out, hidden=(256, 128, 64)):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class GCN(nn.Module):
    def __init__(self, adj_norm: torch.Tensor, inp: int, hid: int = 64):
        super().__init__()
        self.A = adj_norm
        self.l1 = nn.Linear(inp, hid)
        self.l2 = nn.Linear(hid, hid)
        self.pi = nn.Linear(hid, 1)
        self.v = nn.Sequential(nn.Linear(hid, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):                      # x: (B, N, F)
        h = torch.relu(self.A @ self.l1(x))
        h = torch.relu(self.A @ self.l2(h))
        return self.pi(h).squeeze(-1), self.v(h.mean(dim=1)).squeeze(-1)


def norm_adj(sub) -> torch.Tensor:
    n = sub.n
    A = np.eye(n)
    for u, v in sub.topo.edges:
        A[u, v] = A[v, u] = 1.0
    d = A.sum(1)
    Dm = np.diag(1.0 / np.sqrt(d))
    return torch.tensor(Dm @ A @ Dm, dtype=torch.float32)


def _state_tensor(f, g):
    gg = np.repeat(g[None, :], f.shape[0], axis=0)
    return np.concatenate([f, gg], axis=1)


# ---------------------------------------------------------------------------------------- agents
class BaseAgent:
    name = "base"

    def __init__(self, n_nodes: int, seed: int = 0):
        self.N = n_nodes
        self.training = False
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.train_time_s = 0.0
        self.t_load = 0.0

    def act(self, f, g, mask, explore):  # -> (action, extra)
        raise NotImplementedError

    def place(self, inst: Instance, router) -> dict:
        ep = Episode(inst, router)
        t_pre = t_inf = 0.0
        t0 = time.perf_counter()
        traj = []
        while not ep.done:
            a0 = time.perf_counter()
            f, g = ep.features()
            m = ep.mask()
            a1 = time.perf_counter()
            t_pre += a1 - a0
            if not m.any():
                ep.done = ep.failed = True
                if traj:
                    traj[-1]["r"] -= FAIL_PENALTY
                    traj[-1]["done"] = True
                break
            j, extra = self.act(f, g, m, explore=self.training)
            t_inf += time.perf_counter() - a1
            r = ep.step(int(j))
            traj.append(dict(f=f, g=g, m=m, a=int(j), r=r, done=ep.done, **extra))
        if self.training and traj:
            self.observe(traj)
        res = dict(t_pre=t_pre, t_inf=t_inf, t_solve=time.perf_counter() - t0)
        if ep.failed or len(ep.servers) != inst.n:
            res.update(servers=None, paths=None, status="drl_fail")
        else:
            res.update(servers=ep.servers, paths=ep.paths, status="ok")
        return res

    def observe(self, traj):
        pass


class DDPGAgent(BaseAgent):
    name = "DDPG"

    def __init__(self, n_nodes, seed=0, lr_a=1e-4, lr_c=1e-3, gamma=0.99, tau=0.005, batch=128, buf=50000,
                 noise=0.3):
        super().__init__(n_nodes, seed)
        dim = n_nodes * N_FEAT + N_GLOBAL
        self.actor, self.critic = mlp(dim, n_nodes), mlp(dim + n_nodes, 1)
        self.actor_t, self.critic_t = mlp(dim, n_nodes), mlp(dim + n_nodes, 1)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.oa = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.oc = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.gamma, self.tau, self.batch, self.noise = gamma, tau, batch, noise
        self.buf_size = buf
        self.buf = dict(s=np.zeros((buf, dim), np.float32), a=np.zeros(buf, np.int64), r=np.zeros(buf, np.float32),
                        s2=np.zeros((buf, dim), np.float32), m2=np.zeros((buf, n_nodes), bool),
                        d=np.zeros(buf, np.float32))
        self.ptr = self.size = 0

    @staticmethod
    def _flat(f, g):
        return np.concatenate([f.ravel(), g]).astype(np.float32)

    def act(self, f, g, mask, explore):
        with torch.no_grad():
            logits = self.actor(torch.from_numpy(self._flat(f, g))[None])[0].numpy()
        if explore:
            logits = logits + self.rng.normal(0, self.noise, len(logits))
        logits = np.where(mask, logits, -np.inf)
        return int(np.argmax(logits)), {}

    def observe(self, traj):
        for k, tr in enumerate(traj):
            s = self._flat(tr["f"], tr["g"])
            if k + 1 < len(traj):
                s2 = self._flat(traj[k + 1]["f"], traj[k + 1]["g"])
                m2 = traj[k + 1]["m"]
                d = 0.0
            else:
                s2, m2, d = s, tr["m"], 1.0
            p = self.ptr
            self.buf["s"][p], self.buf["a"][p], self.buf["r"][p] = s, tr["a"], tr["r"]
            self.buf["s2"][p], self.buf["m2"][p], self.buf["d"][p] = s2, m2, d
            self.ptr = (p + 1) % self.buf_size
            self.size = min(self.size + 1, self.buf_size)
            if self.size >= self.batch:
                self._update()

    def _soft(self, logits, mask):
        return torch.softmax(logits.masked_fill(~mask, -1e9), dim=-1)

    def _update(self):
        idx = self.rng.integers(0, self.size, self.batch)
        s = torch.from_numpy(self.buf["s"][idx])
        a = Fnn.one_hot(torch.from_numpy(self.buf["a"][idx]), self.N).float()
        r = torch.from_numpy(self.buf["r"][idx])
        s2 = torch.from_numpy(self.buf["s2"][idx])
        m2 = torch.from_numpy(self.buf["m2"][idx])
        d = torch.from_numpy(self.buf["d"][idx])
        with torch.no_grad():
            a2 = self._soft(self.actor_t(s2), m2)
            y = r + self.gamma * (1 - d) * self.critic_t(torch.cat([s2, a2], 1)).squeeze(-1)
        q = self.critic(torch.cat([s, a], 1)).squeeze(-1)
        lc = Fnn.mse_loss(q, y)
        self.oc.zero_grad()
        lc.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.oc.step()
        m = s[:, :self.N * N_FEAT].reshape(-1, self.N, N_FEAT)[:, :, 3] > 0.5
        pa = self._soft(self.actor(s), m)
        la = -self.critic(torch.cat([s, pa], 1)).mean()
        self.oa.zero_grad()
        la.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.oa.step()
        with torch.no_grad():
            for net, tgt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
                for p_, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p_)

    def state_dict(self):
        return dict(kind="DDPG", N=self.N, actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    train_time_s=self.train_time_s)


class PPOAgent(BaseAgent):
    name = "GCN-PPO"

    def __init__(self, sub, seed=0, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, ent=0.01, epochs=4, rollout=1024,
                 minibatch=256):
        super().__init__(sub.n, seed)
        self.adj = norm_adj(sub)
        self.net = GCN(self.adj, N_FEAT + N_GLOBAL)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip, self.ent = gamma, lam, clip, ent
        self.epochs, self.rollout, self.minibatch = epochs, rollout, minibatch
        self.mem = []

    def act(self, f, g, mask, explore):
        x = torch.from_numpy(_state_tensor(f, g))[None]
        with torch.no_grad():
            logits, v = self.net(x)
        logits = logits[0].masked_fill(~torch.from_numpy(mask), -1e9)
        if explore:
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            return int(a), dict(logp=float(dist.log_prob(a)), v=float(v[0]))
        return int(torch.argmax(logits)), {}

    def observe(self, traj):
        self.mem.append(traj)
        if sum(len(t) for t in self.mem) >= self.rollout:
            self._update()
            self.mem = []

    def _update(self):
        X, M, A, LP, ADV, RET = [], [], [], [], [], []
        for traj in self.mem:
            vals = [t["v"] for t in traj] + [0.0]
            gae = 0.0
            advs = []
            for k in reversed(range(len(traj))):
                done = traj[k]["done"]
                delta = traj[k]["r"] + self.gamma * vals[k + 1] * (0.0 if done else 1.0) - vals[k]
                gae = delta + self.gamma * self.lam * (0.0 if done else 1.0) * gae
                advs.append(gae)
            advs.reverse()
            for k, t in enumerate(traj):
                X.append(_state_tensor(t["f"], t["g"]))
                M.append(t["m"])
                A.append(t["a"])
                LP.append(t["logp"])
                ADV.append(advs[k])
                RET.append(advs[k] + vals[k])
        X = torch.tensor(np.array(X))
        M = torch.tensor(np.array(M))
        A = torch.tensor(A)
        LP = torch.tensor(LP, dtype=torch.float32)
        ADV = torch.tensor(ADV, dtype=torch.float32)
        RET = torch.tensor(RET, dtype=torch.float32)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(A)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.minibatch):
                b = perm[s:s + self.minibatch]
                logits, v = self.net(X[b])
                logits = logits.masked_fill(~M[b], -1e9)
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(A[b])
                ratio = torch.exp(lp - LP[b])
                s1, s2 = ratio * ADV[b], torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ADV[b]
                loss = -torch.min(s1, s2).mean() + 0.5 * Fnn.mse_loss(v, RET[b]) - self.ent * dist.entropy().mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

    def state_dict(self):
        return dict(kind="GCN-PPO", N=self.N, net=self.net.state_dict(), adj=self.adj,
                    train_time_s=self.train_time_s)


def save_agent(agent, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), path)


class _Shell:
    """Minimal holder used to rebuild a PPO agent without the substrate object."""


def load_agent(path: str | Path):
    sd = torch.load(path, weights_only=False)
    if sd["kind"] == "DDPG":
        # Evaluation only needs the actor: no critic, target networks, optimisers or replay buffer.
        # (Constructing the optimisers triggers a one-time import inside PyTorch that takes several
        # seconds and would otherwise be counted as model-loading time.)
        ag = DDPGAgent.__new__(DDPGAgent)
        BaseAgent.__init__(ag, sd["N"])
        ag.actor = mlp(sd["N"] * N_FEAT + N_GLOBAL, sd["N"])
        ag.actor.load_state_dict(sd["actor"])
        ag.noise = 0.0
    else:
        ag = PPOAgent.__new__(PPOAgent)
        BaseAgent.__init__(ag, sd["N"])
        ag.adj = sd["adj"]
        ag.net = GCN(ag.adj, N_FEAT + N_GLOBAL)
        ag.net.load_state_dict(sd["net"])
    ag.train_time_s = sd.get("train_time_s", 0.0)
    ag.training = False
    return ag
