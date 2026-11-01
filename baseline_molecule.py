
pass
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_add_pool
from torch.utils.data import Subset
from models.GIN_BBBP import GIN
from molecule_test_sampling import (
    BOOTSTRAP_SEEDS, aggregate_group_metrics, split_domain_indices,
    TEST_GROUP_SIZE,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ENV_PYTORCH_SEED = 42


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(name):
    if name == "BBBP":
        from baseline_BBBP import BBBPDataset
        return BBBPDataset(str(DATA / "bbbp"), str(DATA / "bbbp" / "raw" / "BBBP.csv"))
    if name == "HIV":
        from baseline_HIV import HIVDataset
        return HIVDataset(str(DATA / "hiv"), str(DATA / "hiv" / "raw" / "HIV.csv"))
    if name == "MUTAG":
        from baseline_mutag import MutagDatasetWithSubstructures
        return MutagDatasetWithSubstructures(root=str(DATA / "MUTAG"))
    raise ValueError(name)


def domain_id(data):
    if hasattr(data, "domain"):
        v = data.domain.detach().cpu().reshape(-1).tolist()
        if len(v) == 1:
            return int(v[0])
        return int(v[0]) * 2 + int(v[1])
    c = int(data.has_carbonyl.item())
    h = int(data.has_halogen.item())
    return c * 2 + h


def split_domains(dataset, seed, bootstrap_mutag=False):
    rng = random.Random(seed)
    domains = {i: [] for i in range(4)}
    for i in range(len(dataset)):
        data = dataset[i]
        if bootstrap_mutag:
            carbonyl = int(data.has_carbonyl.item())
            halogen = int(data.has_halogen.item())


            domain = (0 if carbonyl and halogen else
                      1 if carbonyl else 2 if halogen else 3)
        else:
            domain = domain_id(data)
        domains[domain].append(i)
    if bootstrap_mutag:


        shared_splits = split_domain_indices(domains, split_seed=42)
        return domains, shared_splits
    splits = {}
    for d, ids in domains.items():
        rng.shuffle(ids)
        if len(ids) < 3:
            splits[d] = {"train": ids, "val": ids, "test": ids}
            continue
        n_test = max(1, int(round(0.15 * len(ids))))
        n_val = max(1, int(round(0.15 * len(ids))))
        splits[d] = {
            "train": ids[:len(ids) - n_val - n_test],
            "val": ids[len(ids) - n_val - n_test:len(ids) - n_test],
            "test": ids[len(ids) - n_test:],
        }
    return domains, splits


def batch_forward(model, data, task):
    return model(data, task)


def labels_from(data):
    return data.y.float().reshape(-1)


class GINFeatureModel(nn.Module):
    def __init__(self, in_channels, hidden=64, dropout=0.5):
        super().__init__()
        self.encoder = GIN(
            in_channels=in_channels, hidden_dim=hidden, output_dim=hidden,
            num_layers=3, dropout=dropout
        )
        self.head = nn.Linear(hidden, 1)

    def encode(self, data):
        z, _ = self.encoder(data)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        return z

    def forward(self, data, task=None):
        z = self.encode(data)
        return z, self.head(z).reshape(-1)


class FeatureReplay:
    def __init__(self, max_items=2048):
        self.max_items = max_items
        self.items = []

    def add(self, z, y):
        z = z.detach().cpu()
        y = y.detach().cpu()
        for x, label in zip(z, y):
            self.items.append((x, label.reshape(())))
        self.items = self.items[-self.max_items:]

    def __len__(self):
        return len(self.items)

    def sample(self, n, device):
        if not self.items:
            return None, None
        take = [self.items[random.randrange(len(self.items))] for _ in range(n)]
        z = torch.stack([x for x, _ in take]).to(device)
        y = torch.stack([y for _, y in take]).float().to(device)
        return z, y


class TMNModel(GINFeatureModel):
    def __init__(self, in_channels, hidden=64, dropout=0.5):
        super().__init__(in_channels, hidden, dropout)
        self.discriminator = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )


class EWC:
    def __init__(self):
        self.fisher = {}
        self.anchor = {}

    def penalty(self, model):
        value = torch.zeros((), device=next(model.parameters()).device)
        for name, p in model.named_parameters():
            if name in self.fisher:
                value = value + (self.fisher[name].to(p.device) *
                                 (p - self.anchor[name].to(p.device)).pow(2)).sum()
        return value

    def consolidate(self, model, loader, device):
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        model.eval()
        count = 0
        for data in loader:
            data = data.to(device)
            model.zero_grad(set_to_none=True)
            _, logit = model(data)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, labels_from(data))
            loss.backward()
            count += 1
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)
        if count:
            for n in fisher:
                fisher[n] /= count
        self.fisher = fisher
        self.anchor = {n: p.detach().clone() for n, p in model.named_parameters()}


class Adapter(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, rank), nn.GELU(),
                                 nn.Linear(rank, dim))

    def forward(self, x):
        return self.net(x)


class HidePETModel(nn.Module):
    def __init__(self, in_channels, hidden=64, rank=8, dropout=0.5):
        super().__init__()
        self.backbone = GIN(
            in_channels=in_channels, hidden_dim=hidden, output_dim=hidden,
            num_layers=3, dropout=dropout
        )
        self.shared = Adapter(hidden, rank)
        self.adapters = nn.ModuleList()
        self.head = nn.Linear(hidden, 1)
        self.active_task = 0

    def add_task(self):
        self.adapters.append(Adapter(self.head.in_features, 8))
        self.active_task = len(self.adapters) - 1

    def set_task(self, task):
        while len(self.adapters) <= task:
            self.add_task()
        self.active_task = task
        for adapter in self.adapters:
            for p in adapter.parameters():
                p.requires_grad = adapter is self.adapters[task]

    def set_backbone_trainable(self, enabled):
        for p in self.backbone.parameters():
            p.requires_grad = enabled

    def forward(self, data, task=None):
        task = self.active_task if task is None else task
        z, _ = self.backbone(data)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        z = z + self.shared(z) + self.adapters[task](z)
        return z, self.head(z).reshape(-1)


class WeightedGINConv(MessagePassing):
    pass
    def __init__(self, hidden):
        super().__init__(aggr="add")
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.shape[1])
        aggregate = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return self.mlp((1.0 + self.eps) * x + aggregate)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.reshape(-1, 1)


class GINNodeEncoder(nn.Module):
    pass
    def __init__(self, in_channels, hidden=64, num_layers=3, dropout=0.5):
        super().__init__()
        self.input = nn.Linear(in_channels, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(WeightedGINConv(hidden))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout

    def forward(self, data, edge_weight=None):
        x = self.input(data.x.float())
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, data.edge_index, edge_weight=edge_weight)


            if x.shape[0] > 1:
                x = norm(x)
            x = F.relu(x) + residual
            x = F.dropout(x, p=self.dropout, training=self.training)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return x, batch, global_add_pool(x, batch)


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


class EMoNetModel(nn.Module):
    pass
    def __init__(self, in_channels, hidden=64, dropout=0.5,
                 environment_prior=0.5, grl_scale=1.0):
        super().__init__()
        self.encoder = GINNodeEncoder(in_channels, hidden, dropout=dropout)
        self.environment_gate = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)
        )
        self.environment_predictor = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.environment_alignment = nn.Linear(hidden, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.environment_prior = float(environment_prior)
        self.grl_scale = float(grl_scale)

    def forward(self, data, task=None, return_aux=False):
        h, batch, _ = self.encoder(data)
        src, dst = data.edge_index
        edge_probability = torch.sigmoid(self.environment_gate((
            h[src] + h[dst]
        ) / 2.0)).reshape(-1)
        if self.training:

            uniform = torch.rand_like(edge_probability).clamp(1e-6, 1.0 - 1e-6)
            edge_gate = torch.sigmoid((
                torch.logit(edge_probability.clamp(1e-6, 1.0 - 1e-6)) +
                torch.log(uniform) - torch.log1p(-uniform)
            ) / 0.67)
        else:
            edge_gate = edge_probability
        env_nodes, _, _ = self.encoder(data, edge_weight=edge_gate)
        inv_nodes, _, _ = self.encoder(data, edge_weight=1.0 - edge_gate)
        z_env = global_add_pool(env_nodes, batch)
        z_inv = global_add_pool(inv_nodes, batch)


        z = z_inv + z_env * torch.sigmoid(self.environment_alignment(z_env))
        logits = self.classifier(z).reshape(-1)
        env_logits = self.environment_predictor(
            _GradientReverse.apply(z_env, self.grl_scale)
        ).reshape(-1)
        prior = self.environment_prior
        eps = 1e-6
        p = edge_probability.clamp(eps, 1.0 - eps)
        kl = p * (p.log() - np.log(prior)) + (1.0 - p) * (
            (1.0 - p).log() - np.log(1.0 - prior)
        )
        aux = {"environment_logits": env_logits, "gate_kl": kl.mean()}
        return (z, logits, aux) if return_aux else (z, logits)


class NeGoModel(nn.Module):
    pass
    def __init__(self, in_channels, hidden=64, dropout=0.5, num_classes=2):
        super().__init__()
        self.encoder = GINNodeEncoder(in_channels, hidden, dropout=dropout)
        self.prompts = nn.Parameter(torch.randn(num_classes, hidden) * 0.02)
        self.prompt_attention = nn.MultiheadAttention(
            hidden, num_heads=4 if hidden % 4 == 0 else 1,
            dropout=dropout, batch_first=True
        )
        self.environment_projection = nn.Linear(hidden, hidden)
        self.node_mask = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.num_classes = num_classes

    def forward(self, data, task=None, return_aux=False):
        h, batch, z_graph = self.encoder(data)
        per_graph_answers = []
        for graph_id in range(z_graph.shape[0]):
            nodes = h[batch == graph_id].unsqueeze(0)
            q = self.prompts.unsqueeze(0)
            answer, _ = self.prompt_attention(q, nodes, nodes, need_weights=False)
            per_graph_answers.append(answer.squeeze(0))
        answers = torch.stack(per_graph_answers, dim=0)
        prompt_scores = F.cosine_similarity(
            z_graph.unsqueeze(1), answers, dim=-1
        )


        env_context = answers.mean(dim=1)
        node_context = env_context[batch]
        decoded = h + self.environment_projection(node_context)
        mask = torch.sigmoid(self.node_mask(torch.cat([h, decoded], dim=-1)))
        z_causal = global_add_pool(h * mask, batch)
        logits = self.classifier(z_causal).reshape(-1)
        aux = {"negative_logits": prompt_scores, "node_mask": mask}
        return (z_causal, logits, aux) if return_aux else (z_causal, logits)


def make_loaders(dataset, splits, batch_size, bootstrap_mutag=False):
    loaders = {}
    for d in range(4):
        loaders[d] = {
            k: DataLoader(Subset(dataset, splits[d][k]), batch_size=batch_size,
                          shuffle=(k == "train"))
            for k in ("train", "val")
        }
        if bootstrap_mutag:
            loaders[d]["test"] = [
                DataLoader(Subset(dataset, splits[d]["test_groups"][seed]),
                           batch_size=batch_size, shuffle=False)
                for seed in BOOTSTRAP_SEEDS
            ]
            loaders[d]["test_seeds"] = list(BOOTSTRAP_SEEDS)
        else:
            loaders[d]["test"] = DataLoader(
                Subset(dataset, splits[d]["test"]), batch_size=batch_size,
                shuffle=False
            )
    return loaders


def evaluate(model, loader, device, task):
    model.eval()
    ys, ps, loss_sum, count = [], [], 0.0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, logit = batch_forward(model, data, task)
            y = labels_from(data)
            loss_sum += nn.functional.binary_cross_entropy_with_logits(
                logit, y, reduction="sum"
            ).item()
            count += y.numel()
            ys.extend(y.cpu().tolist())
            ps.extend(torch.sigmoid(logit).cpu().tolist())
    if not count:
        return {"loss": None, "accuracy": None, "auc": None, "n": 0}
    pred = [int(x >= 0.5) for x in ps]
    acc = sum(int(a == b) for a, b in zip(pred, ys)) / len(ys)
    try:
        auc = float(roc_auc_score(ys, ps))
    except ValueError:
        auc = None
    return {"loss": loss_sum / count, "accuracy": acc, "auc": auc, "n": count}


def evaluate_test(model, test_loaders, device, task, seeds=None):
    pass
    if not isinstance(test_loaders, list):
        return evaluate(model, test_loaders, device, task)
    seeds = list(seeds or BOOTSTRAP_SEEDS)
    group_results = [evaluate(model, loader, device, task)
                     for loader in test_loaders]
    return {
        "bootstrap_test": {
            "seeds": seeds,
            "samples_per_group": group_results[0]["n"] if group_results else 0,
            "groups": group_results,
            "summary": aggregate_group_metrics(group_results),
        }
    }


def train_tmn(model, loaders, splits, device, epochs, lr, ewc_lambda):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    ewc, replay = EWC(), FeatureReplay()
    results = {}
    for task in range(4):
        model.train()
        for _ in range(epochs):
            for data in loaders[task]["train"]:
                data = data.to(device)
                z, logit = model(data)
                y = labels_from(data)
                loss = criterion(logit, y)
                if len(replay):
                    old_z, old_y = replay.sample(len(y), device)
                    loss = loss + criterion(model.head(old_z).reshape(-1), old_y)
                    real_d = model.discriminator(z.detach()).reshape(-1)
                    old_d = model.discriminator(old_z.detach()).reshape(-1)
                    loss = loss + 0.25 * (
                        criterion(real_d, torch.ones_like(real_d)) +
                        criterion(old_d, torch.zeros_like(old_d))
                    )
                if ewc.fisher:
                    loss = loss + ewc_lambda * ewc.penalty(model)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        model.eval()
        with torch.no_grad():
            for data in loaders[task]["train"]:
                data = data.to(device)
                z, _ = model(data)
                replay.add(z, labels_from(data))
        ewc.consolidate(model, loaders[task]["train"], device)
        results[f"after_domain_{task}"] = {
            str(d): evaluate_test(
                model, loaders[d]["test"], device, None,
                loaders[d].get("test_seeds")
            )
            for d in range(task + 1)
        }
    return results


def train_hidepet(model, loaders, device, epochs, lr):
    results = {}
    for task in range(4):
        model.set_task(task)
        model.set_backbone_trainable(task == 0)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            model.train()
            for data in loaders[task]["train"]:
                data = data.to(device)
                _, logit = model(data, task)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logit, labels_from(data)
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
        model.set_backbone_trainable(False)
        results[f"after_domain_{task}"] = {
            str(d): evaluate_test(
                model, loaders[d]["test"], device, d,
                loaders[d].get("test_seeds")
            )
            for d in range(task + 1)
        }
        for p in model.adapters[task].parameters():
            p.requires_grad = False
    return results


def train_environment_baseline(model, loaders, device, epochs, lr, method):
    pass
    results = {}
    for task in range(4):
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            model.train()
            for data in loaders[task]["train"]:
                data = data.to(device)
                y = labels_from(data)
                _, logits, aux = model(data, task, return_aux=True)
                task_loss = F.binary_cross_entropy_with_logits(logits, y)
                if method == "EMoNet":
                    env_loss = F.binary_cross_entropy_with_logits(
                        aux["environment_logits"], y
                    )
                    loss = task_loss + 0.2 * env_loss + 0.01 * aux["gate_kl"]
                else:


                    negative_target = (1.0 - y).long().clamp(0, 1)
                    negative_loss = F.cross_entropy(
                        aux["negative_logits"], negative_target
                    )
                    loss = task_loss + negative_loss + 0.01 * aux["node_mask"].mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        results[f"after_domain_{task}"] = {
            str(d): evaluate_test(
                model, loaders[d]["test"], device, d,
                loaders[d].get("test_seeds")
            )
            for d in range(task + 1)
        }
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["BBBP", "HIV", "MUTAG"])
    p.add_argument("--baseline", required=True,
                   choices=["tmn", "hidepet", "EMoNet", "NeGo"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ewc-lambda", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(args.dataset)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")
    bootstrap_mutag = args.dataset == "MUTAG"
    domain_indices, splits = split_domains(dataset, args.seed, bootstrap_mutag)
    loaders = make_loaders(dataset, splits, args.batch_size, bootstrap_mutag)
    in_channels = int(dataset[0].x.shape[1])
    if args.baseline == "tmn":
        model = TMNModel(in_channels, args.hidden_dim).to(device)
        results = train_tmn(model, loaders, splits, device, args.epochs,
                            args.lr, args.ewc_lambda)
    elif args.baseline == "hidepet":
        model = HidePETModel(in_channels, args.hidden_dim).to(device)
        results = train_hidepet(model, loaders, device, args.epochs, args.lr)
    elif args.baseline == "EMoNet":
        model = EMoNetModel(in_channels, args.hidden_dim).to(device)
        results = train_environment_baseline(
            model, loaders, device, args.epochs, args.lr, "EMoNet"
        )
    else:
        model = NeGoModel(in_channels, args.hidden_dim).to(device)
        results = train_environment_baseline(
            model, loaders, device, args.epochs, args.lr, "NeGo"
        )
    output = Path(args.output) if args.output else (
        ROOT / "baseline_results" /
        f"{args.dataset.lower()}_{args.baseline}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "dataset": args.dataset, "baseline": args.baseline,
        "device": str(device), "epochs": args.epochs,
        "test_protocol": ({
            "method": "bootstrap_with_replacement",
            "groups": len(BOOTSTRAP_SEEDS),
            "seeds": list(BOOTSTRAP_SEEDS),
            "samples_per_group": TEST_GROUP_SIZE,
            "test_source": "held-out domain test pool",
        } if bootstrap_mutag else "standard_holdout"),
        "test_group_indices": ({
            str(d): {str(seed): ids for seed, ids in
                     splits[d]["test_groups"].items()}
            for d in range(4)
        } if bootstrap_mutag else None),
        "domain_counts": {str(d): len(domain_indices[d])
                          for d in range(4)},
        "results": results,
    }, indent=2))
    print(json.dumps({"dataset": args.dataset, "baseline": args.baseline,
                      "output": str(output), "results": results}, indent=2))


if __name__ == "__main__":
    main()
