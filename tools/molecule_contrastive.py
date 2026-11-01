pass

import random

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool


def extract_gin_graph_embeddings(model, graph_batch):
    pass
    x = model.initial_lin(graph_batch.x)
    for layer in range(model.num_layers):
        residual = x
        x = model.convs1[layer](x, graph_batch.edge_index)
        x = model.convs2[layer](x, graph_batch.edge_index)
        x = F.relu(x) + residual
    x = global_add_pool(x, graph_batch.batch)
    x = model.classifier1(x)
    return model.Relu(x)


def _supcon_loss(features, labels, temperature):
    similarity = torch.matmul(features, features.T) / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()
    count = features.size(0)
    non_self = ~torch.eye(count, dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None, :]) & non_self
    log_denominator = torch.logsumexp(
        similarity.masked_fill(~non_self, torch.finfo(similarity.dtype).min), dim=1
    )
    log_probability = similarity - log_denominator[:, None]
    positive_count = positives.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return None
    per_anchor = -(log_probability.masked_fill(~positives, 0).sum(dim=1)
                   / positive_count.clamp_min(1))
    return per_anchor[valid].mean()


def compute_min_contrastive_distance(
    model,
    dataset,
    domain_train_indices,
    query_domain,
    batch_size,
    device,
    max_per_class=64,
    projection_epochs=30,
    temperature=0.1,
    seed=42,
):
    pass
    sampled = {}
    sampler = random.Random(seed)
    for domain, indices in domain_train_indices.items():
        by_label = {}
        for index in indices:
            label_tensor = torch.as_tensor(dataset[index].y).detach().cpu().reshape(-1)
            if label_tensor.numel() != 1 or not torch.isfinite(label_tensor[0]):
                continue
            label = float(label_tensor[0].item())
            by_label.setdefault(label, []).append(index)
        selected = []
        for label_indices in by_label.values():
            if len(label_indices) > max_per_class:
                selected.extend(sampler.sample(label_indices, max_per_class))
            else:
                selected.extend(label_indices)
        if selected:
            sampled[domain] = selected

    if query_domain not in sampled or len(sampled) < 2:
        return float("inf"), None

    embedding_rows, label_rows, domain_rows = [], [], []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for domain, indices in sampled.items():
            graphs = [dataset[index] for index in indices]
            loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
            for graph_batch in loader:
                graph_batch = graph_batch.to(device)
                embeddings = extract_gin_graph_embeddings(model, graph_batch)
                labels = graph_batch.y.reshape(-1).to(device=device, dtype=torch.float32)
                finite = torch.isfinite(labels)
                if finite.any():
                    embedding_rows.append(embeddings[finite])
                    label_rows.append(labels[finite])
                    domain_rows.extend([domain] * int(finite.sum().item()))
    model.train(was_training)

    if not embedding_rows:
        return float("inf"), None
    embeddings = torch.cat(embedding_rows, dim=0).detach()
    labels = torch.cat(label_rows, dim=0)
    if embeddings.size(0) < 2:
        return float("inf"), None

    cuda_devices = []
    if torch.device(device).type == "cuda":
        cuda_devices = [torch.device(device).index or torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(seed)
        projection = nn.Sequential(
            nn.Linear(embeddings.size(1), embeddings.size(1)),
            nn.ReLU(),
            nn.Linear(embeddings.size(1), embeddings.size(1)),
        ).to(device)
        optimizer = torch.optim.Adam(projection.parameters(), lr=1e-3)
        projection.train()
        for _ in range(projection_epochs):
            optimizer.zero_grad()
            projected = F.normalize(projection(embeddings), dim=1)
            loss = _supcon_loss(projected, labels, temperature)
            if loss is None:
                break
            loss.backward()
            optimizer.step()

        projection.eval()
        with torch.no_grad():
            projected = projection(embeddings)

    domain_rows = torch.as_tensor(domain_rows, device=device)
    prototypes = {}
    for domain in sampled:
        mask = domain_rows == domain
        if mask.any():
            prototypes[domain] = projected[mask].mean(dim=0)

    if query_domain not in prototypes:
        return float("inf"), None
    distances = {
        domain: torch.linalg.vector_norm(prototype - prototypes[query_domain]).item()
        for domain, prototype in prototypes.items()
        if domain != query_domain
    }
    if not distances:
        return float("inf"), None
    nearest_domain = min(distances, key=distances.get)
    return distances[nearest_domain], nearest_domain
