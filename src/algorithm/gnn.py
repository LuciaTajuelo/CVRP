from typing import List, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models import Solution


class SimpleGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation=F.relu):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.self_lin = nn.Linear(in_dim, out_dim, bias=True)
        self.act = activation

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_self = self.self_lin(x)
        h_neigh = self.lin(x)
        if adj is None:
            agg = h_neigh
        else:
            deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
            agg = torch.matmul(adj, h_neigh) / deg
        return self.act(h_self + agg)

class GNNEncoder(nn.Module):
    def __init__(self, node_feat_dim: int, hid_dim: int = 256, n_layers: int = 5):
        super().__init__()
        layers = []
        in_dim = node_feat_dim
        for _ in range(n_layers):
            layers.append(SimpleGraphConv(in_dim, hid_dim))
            in_dim = hid_dim
        self.layers = nn.ModuleList(layers)
        self.hid_dim = hid_dim

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x
        for l in self.layers:
            h = l(h, adj)
        return h

def two_opt_guided(route, dist, demands, capacity, node_emb):
    best = route
    improved = True

    while improved:
        improved = False

        candidates = []
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                candidates.append((i, j))

        # Priorizar swaps con nodos disimilares en embedding
        candidates.sort(
            key=lambda ij: torch.norm(
                node_emb[best[ij[0]]] - node_emb[best[ij[1]]]
            ),
            reverse=True,
        )

        for i, j in candidates:
            new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
            load = sum(demands[n] for n in new_route if n != 0)
            if load > capacity:
                continue

            old_cost = sum(dist[best[k], best[k + 1]] for k in range(len(best) - 1))
            new_cost = sum(dist[new_route[k], new_route[k + 1]] for k in range(len(new_route) - 1))

            if new_cost < old_cost:
                best = new_route
                improved = True
                break

    return best

class BeamDecoder(nn.Module):
    def __init__(self, hid_dim, beam_width=1, alpha=0.5, beta=0.1):
        super().__init__()
        self.query_net = nn.Linear(hid_dim + 2, hid_dim)
        self.key_net = nn.Linear(hid_dim, hid_dim)
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta

    def score(self, context: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        q = self.query_net(context)
        k = self.key_net(keys)
        return torch.matmul(k, q) / math.sqrt(q.shape[0])

    def forward(self, node_emb, demands, capacity, depot_idx, coords=None):
        N = node_emb.size(0)
        device = node_emb.device
        depot = int(depot_idx)

        dist = torch.cdist(coords, coords) if coords is not None else None

        remaining = set(range(N))
        remaining.remove(depot)
        routes = []

        while remaining:
            route = [depot]
            load = capacity
            last = depot

            while True:
                candidates = [c for c in remaining if demands[c] <= load]
                if not candidates:
                    break

                last_emb = node_emb[last]
                avgd = 0.0
                if dist is not None:
                    avgd = dist[last][candidates].mean() / dist.max().clamp(min=1e-6)

                context = torch.cat([
                    last_emb,
                    torch.tensor([load / capacity, avgd], device=device)
                ], dim=0)

                gnn_scores = self.score(context, node_emb[candidates])
                greedy_scores = -dist[last][candidates] if dist is not None else 0.0

                scores = self.alpha * gnn_scores + self.beta * greedy_scores
                nxt = candidates[int(torch.argmax(scores))]

                route.append(nxt)
                remaining.remove(nxt)
                load -= float(demands[nxt])
                last = nxt

            route.append(depot)
            if dist is not None:
                route = two_opt_guided(route, dist, demands, capacity, node_emb)
            routes.append(route)

        return routes

class CVRPGNNSolver(nn.Module):
    def __init__(self, node_feat_dim=6, hid_dim=256, encoder_layers=5, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.encoder = GNNEncoder(node_feat_dim, hid_dim, encoder_layers).to(self.device)
        self.decoder = BeamDecoder(hid_dim).to(self.device)

    @staticmethod
    def build_node_features(coords, demands, depot, capacity):
        N = coords.shape[0]
        feats = coords.float()
        feats = torch.cat([feats, (demands / capacity).unsqueeze(1)], dim=1)

        is_depot = torch.zeros((N, 1), device=coords.device)
        is_depot[depot] = 1.0
        feats = torch.cat([feats, is_depot], dim=1)

        dist_dep = torch.norm(coords - coords[depot], dim=1, keepdim=True)
        feats = torch.cat([feats, dist_dep / dist_dep.max().clamp(min=1e-6)], dim=1)

        idx_feat = torch.arange(N, device=coords.device).float().unsqueeze(1) / (N - 1)
        feats = torch.cat([feats, idx_feat], dim=1)

        return feats

    def solve(self, instance: Dict):
        coords = torch.tensor(instance['coords'], device=self.device)
        demands = torch.tensor(instance['demands'], device=self.device)
        capacity = float(instance['capacity'])
        depot = int(instance.get('depot', 0))

        feats = self.build_node_features(coords, demands, depot, capacity)

        with torch.no_grad():
            paird = torch.cdist(coords, coords)
            sigma = paird.mean().clamp(min=1e-3)
            adj = torch.exp(-(paird ** 2) / (2 * sigma ** 2))
            adj.fill_diagonal_(0.0)

        node_emb = self.encoder(feats, adj)
        return self.decoder(node_emb, demands, capacity, depot, coords)

# ==========================
#  Wrapper
# ==========================
def gnn_beam_guided(instance, beam_width=1, alpha=0.5, beta=0.1, sigma=0.1):
    solver = CVRPGNNSolver()
    solver.decoder.beam_width = beam_width
    solver.decoder.alpha = alpha
    solver.decoder.beta = beta

    # Construir instancia interna
    inst = {
        'coords': [instance.depot] + [c.coords() for c in instance.customers],
        'demands': [0] + [c.demand for c in instance.customers],
        'capacity': instance.vehicle_capacity,
        'depot': 0,
    }

    coords = torch.tensor(inst['coords'])
    paird = torch.cdist(coords, coords)
    adj = torch.exp(-(paird ** 2) / (2 * sigma ** 2))
    adj.fill_diagonal_(0.0)

    # Embeddings
    feats = solver.build_node_features(coords, torch.tensor(inst['demands']), 0, inst['capacity'])
    with torch.no_grad():
        node_emb = solver.encoder(feats, adj)

    # Resolver rutas
    routes = solver.decoder(node_emb, torch.tensor(inst['demands']), inst['capacity'], 0, coords)

    # Construir Solution
    solution = Solution(instance)
    for r in routes:
        solution.add_route([n for n in r if n != 0])
    solution.validate()

    return solution
