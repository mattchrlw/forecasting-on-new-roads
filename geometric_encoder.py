import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from graph import generate_quotient_graph, generate_graphs, feature_extract, subgraph
import random
from sklearn.preprocessing import MinMaxScaler

# layers: 4 input for a 40 node subgraph, then 320, then 40 output

# geometric autoencoder
class GeometricEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(160, 320), nn.ReLU(), nn.Linear(320, 40))

    def forward(self, x):
        return self.l1(x)


class GeometricDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(40, 320), nn.ReLU(), nn.Linear(320, 160))

    def forward(self, x):
        return self.l1(x)

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

Q, nearest_node, clusters, gdf_nodes, gdf_edges = generate_quotient_graph()
Q1, Q2 = generate_graphs(Q, nearest_node, clusters, gdf_nodes, gdf_edges)
Q, _ = generate_graphs(Q, nearest_node, clusters, gdf_nodes, gdf_edges, nearest=True)
source = random.choice(list(nearest_node.keys()))
H1, H2 = subgraph(Q1, source), subgraph(Q2, source)
# print(Q, nearest_node, clusters, gdf_nodes, gdf_edges)

scaler = MinMaxScaler()
scaler.fit(feature_extract(Q))

features = feature_extract(Q1)
features_H1, features_H2 = scaler.transform(feature_extract(H1)), scaler.transform(feature_extract(H2))
print(features_H1, features_H2)