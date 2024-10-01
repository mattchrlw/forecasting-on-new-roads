from shapely import MultiPoint
from collections import defaultdict
from scipy import spatial
from pprint import pprint
import shapely
import csv
import osmnx as ox
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import argparse
import time
import pandas as pd
import random
import torch
from sklearn.preprocessing import MinMaxScaler

MODES = ['identity', 'rbf', 'rbf-osm', 'osm', 'osm-length', 'osm-length-speed']

"""
Load the METR-LA dataset's locations.

Output: A dictionary, with each key being a METR-LA node index and the value being the (x, y) coordinate.
"""
def load_metr_la():
    metr_la_nodes = {}

    with open('data/graph_sensor_locations.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            metr_la_nodes[int(row[1])] = (float(row[3]), float(row[2]))

    return metr_la_nodes

"""
Load the METR-LA adjacency matrix.

Output: An adjacency matrix, with order matching the order of indices in the dataset.
"""
def load_metr_la_adj():
    with open('data/adj_METR-LA.pkl', 'rb') as f:
        metr_la_adj = pickle.load(f, encoding='latin1')[2]

    return metr_la_adj

"""
Load the OSM data, given a dictionary of traffic sensor nodes.

Output:
- poly: an undirected graph consisting of the OSM data contained within the convex hull of the given area.
- gdf_nodes: a graph containing information about the traffic nodes.
- gdf_edges: a graph containing information about the traffic edges.
"""
def load_osm(nodes):
    # convex hull
    hull = shapely.convex_hull(MultiPoint(list(nodes.values())))

    poly = ox.graph.graph_from_polygon(
        hull,
        custom_filter='["highway"~"motorway|trunk"]',
        truncate_by_edge=True,
    )

    ox.routing.add_edge_speeds(poly)
    ox.distance.add_edge_lengths(poly)

    # convert to undirected for simplicity
    poly = ox.convert.to_undirected(poly)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(poly)
    return poly, gdf_nodes, gdf_edges

"""
For each METR-LA node, find the *nearest* OSM node.

Input:
- nodes: the METR-LA nodes.
- osm: the OSM graph.

Output:
- nearest_node: a dictionary mapping a METR-LA node to the nearest OSM node.
"""
def find_matching(nodes, osm):
    nearest_node = {}

    for idx, i in nodes.items():
        nearest_node[idx] = ox.distance.nearest_nodes(osm, i[0], i[1])

    return nearest_node

"""
Maps each of the METR-LA nodes to its nearest OSM node's coordinates.

Input:
- gdf_nodes: the GeoDataFrame containing information about the OSM node.
- nearest_node: the dictionary mapping a METR-LA node to an OSM node.

Output: a dictionary mapping the METR-LA node to its nearest OSM node's coordinates.
"""
def map_to_coords(gdf_nodes, nearest_node):
    return {
        k: (float(gdf_nodes.loc[node]['x']), float(gdf_nodes.loc[node]['y']))
        for k, node in nearest_node.items()
    }

"""
A utility function that does some janky maximum thing.
"""
def _max_with_lists(l):
    maximum = 1
    for i in l:
        if type(i) == str:
            maximum = max(int(i), maximum)

    return maximum

"""
This function returns a subset of OSM features related to a given node.
It first gets all the nodes related to a cluster and gets all incoming-outgoing nodes
and then aggregates features related to those nodes.
"""
def add_osm_features(graph, nearest_node, clusters, gdf_nodes, gdf_edges):
    features = {}
    for node in graph.nodes():
        nodes = clusters[nearest_node[node]]
        # u-side and v-side edges (incoming and outgoing)
        u_side_edges = gdf_edges[gdf_edges.index.get_level_values(0).isin(nodes)]
        v_side_edges = gdf_edges[gdf_edges.index.get_level_values(1).isin(nodes)]
        df = pd.concat([u_side_edges, v_side_edges])

        # this aggregates all features
        features[node] = {
            # TODO: make this a sampling instead of an averaging
            "lanes": _max_with_lists(df['lanes'].values),
            "speed_kph": max(df['speed_kph'])
            # add x and y coordinates for "centers"
        }
    
    return gdf_nodes, features

"""
Constructs a quotient graph, collapsing the original OSM graph to a graph with METR-LA nodes.
"""
def quotient_graph(poly, nearest_node, gdf_nodes):
    A = []
    coords_to_node = {}
    clusters = defaultdict(lambda: [])

    for node in set(nearest_node.values()):
        x, y = gdf_nodes.loc[node]['x'], gdf_nodes.loc[node]['y']
        A.append([x, y])
        coords_to_node[f"{np.array([x, y])}"] = node

    # Necessary conversion step.
    A = np.array(A)

    for node in poly.nodes():
        x, y = gdf_nodes.loc[node]['x'], gdf_nodes.loc[node]['y']
        kdt = spatial.KDTree(A).query([x, y])
        nearest = f"{A[kdt[1]]}"
        clusters[coords_to_node[nearest]].append(gdf_nodes.loc[node].name)

    Q = nx.Graph(nx.quotient_graph(poly, partition=clusters))

    # extra nx.Graph wrapper converts this to a regular Graph instead of a MultiGraph
    return Q, clusters

"""
Relabels a quotient graph with the original traffic nodes, adding self edges and filling in missing edges.
"""
def relabel_graph(graph, nearest_node, clusters):
    # clusters matches cluster_origin -> cluster_item[].
    # to generate two random graphs, we need to:
    # - replace cluster_origin -> cluster_item[] with metr_la -> cluster_item[].
    # - construct the right adjacency matrix with metr_la
    # - output this cluster adjacency alongside cluster_item.
    # - relabel with two random cluster_item[]

    node_to_origin = {}

    for k, v in clusters.items():
        for i in v:
            node_to_origin[i] = k

    reversed_nearest_node = {v: k for k, v in nearest_node.items()}
    Q = nx.relabel_nodes(graph, lambda x: reversed_nearest_node[node_to_origin[next(iter(x))]])

    # add self loops
    for node in Q.nodes():
        Q.add_edge(node, node)

    # if there are two traffic nodes that are closest to the same OSM node, construct a list of them.
    duplicates = defaultdict(lambda: set())
    for node_id, i in set(nearest_node.items()):
        for v in nearest_node.values():
            if v == i:
                duplicates[i].add(node_id)

    # add information of the original node to all other nodes.
    for s in duplicates.values():
        if len(s) > 1:
            origin = None
            for node in s:
                if node in Q.nodes():
                    origin = node
                    break
            for node in s:
                if node not in Q.nodes():
                    # add new node
                    Q.add_node(node)
                    # link to other identical nodes
                    Q.add_edge(node, origin)
                    # add all similar edges
                    for edge in Q.edges(origin):
                        Q.add_edge(node, edge[1])

    return nx.Graph(Q)

"""
Generates a length matrix, which gives actual path distance between nodes.
"""
def generate_length_matrix(metr_la, adj, gdf_edges, nearest_node, poly, speed=False):
    length_mat = np.zeros((len(metr_la), len(metr_la)))
    for idx, i in enumerate(metr_la.keys()):
        for jdx, j in enumerate(metr_la.keys()):
            if adj[idx, jdx] > 0:
                path = ox.shortest_path(poly, nearest_node[i], nearest_node[j])
                if len(path) > 2:
                    for k in range(len(path) - 1):
                        u, v = path[k], path[k + 1]
                        try:
                            length = float(gdf_edges.loc[u, v]['length'].iloc[0])
                            if speed:
                                speed = float(gdf_edges.loc[u, v]['speed_kph'].iloc[0])
                                length_mat[idx, jdx] += length / speed
                            else:
                                length_mat[idx, jdx] += length
                        except KeyError:
                            length = float(gdf_edges.loc[v, u]['length'].iloc[0])
                            if speed:
                                speed = float(gdf_edges.loc[v, u]['speed_kph'].iloc[0])
                                length_mat[idx, jdx] += length / speed
                            else:
                                length_mat[idx, jdx] += length

    length_mat_scaled = np.where(length_mat > 0, 1 - length_mat/np.max(length_mat), 0)
    np.fill_diagonal(length_mat_scaled, 1)

    return length_mat_scaled

"""
Generates a "pair" of graphs from one graph.
"""
def generate_graphs(Q, nearest_node, clusters, gdf_nodes, gdf_edges, nearest=False):
    print("generate_graphs nearest", nearest)
    if nearest:
        partition_1 = nearest_node
        partition_2 = partition_1
    else:
        partition_1 = {k: random.choice(clusters[v]) for k, v in nearest_node.items()}
        partition_2 = {k: random.choice(clusters[v]) for k, v in nearest_node.items()}

    coordinates_1 = {k: {
        'x': gdf_nodes.loc[v]['x'], 
        'y': gdf_nodes.loc[v]['y'],
    } for k, v in partition_1.items()}
    coordinates_2 = {k: {
        'x': gdf_nodes.loc[v]['x'],
        'y': gdf_nodes.loc[v]['y']
    } for k, v in partition_2.items()}

    Q1, Q2 = Q.copy(), Q.copy()

    for (_, d) in Q1.nodes(data=True):
        d.clear()
    for (_, d) in Q2.nodes(data=True):
        d.clear()

    for (_, _, d) in Q1.edges(data=True):
        d.clear()
    for (_, _, d) in Q2.edges(data=True):
        d.clear()

    nx.set_node_attributes(Q1, coordinates_1)
    nx.set_node_attributes(Q2, coordinates_2)

    for k, v in partition_1.items():
        try:
            u_side_edges = gdf_edges.xs(v, level='u')
            Q1.nodes[k]['lanes'] = _max_with_lists(u_side_edges['lanes'].values)
            Q1.nodes[k]["speed_kph"] = random.choice(u_side_edges['speed_kph'].values)
        except KeyError:
            v_side_edges = gdf_edges.xs(v, level='v')
            Q1.nodes[k]['lanes'] = _max_with_lists(v_side_edges['lanes'].values)
            Q1.nodes[k]["speed_kph"] = random.choice(v_side_edges['speed_kph'].values)

    for k, v in partition_2.items():
        try:
            u_side_edges = gdf_edges.xs(v, level='u')
            Q2.nodes[k]['lanes'] = _max_with_lists(u_side_edges['lanes'].values)
            Q2.nodes[k]["speed_kph"] = random.choice(u_side_edges['speed_kph'].values)
        except KeyError:
            v_side_edges = gdf_edges.xs(v, level='v')
            Q2.nodes[k]['lanes'] = _max_with_lists(v_side_edges['lanes'].values)
            Q2.nodes[k]["speed_kph"] = random.choice(v_side_edges['speed_kph'].values)

    attributes = set()
    for _, d in Q1.nodes(data=True):
        attributes.update(d.keys())
    for _, d in Q2.nodes(data=True):
        attributes.update(d.keys())
    print("THE ATTRIBUTES ARE", attributes)

    # TODO: scaling the coordinates...
    # nx.add_edge_lengths(Q1.nodes[])

    return Q1, Q2
    # pprint(Q1)

    # fig = plt.figure()
    # plt.axis('off')
    # plt.tight_layout()
    # plt.scatter(*zip(*coordinates_1))
    # fig.savefig(f"coordinates_1.png")

    # fig = plt.figure()
    # plt.axis('off')
    # plt.tight_layout()
    # plt.scatter(*zip(*coordinates_2 ))
    # fig.savefig(f"coordinates_2.png")

"""
Extract features.
"""
def feature_extract(G):
    return torch.tensor(list(map(
        lambda x: [x[1]['x'], x[1]['y'], x[1]['lanes'], x[1]['speed_kph']], 
        G.nodes(data=True)))
    )

"""
Generates the quotient graph.

Q: final networkx graph
nearest_node: mapping between traffic node and nearest OSM node
clusters: clusters of OSM nodes based on traffic node
gdf_nodes: GDF with node features
gdf_edges: GDF with edge features
"""
def generate_quotient_graph():
    metr_la = load_metr_la()
    # {id: (x, y)}
    # generate more: 
    # metr_la = load_metr_la()
    # generate unseen nodes between???
    # more_metr_la = more(metr_la)
    #
    traffic, gdf_nodes, gdf_edges = load_osm(metr_la)
    nearest_node = find_matching(metr_la, traffic)
    Q, clusters = quotient_graph(traffic, nearest_node, gdf_nodes)
    Q = relabel_graph(Q, nearest_node, clusters)

    return Q, nearest_node, clusters, gdf_nodes, gdf_edges

"""
Generates an adjacency matrix.
"""
def generate_adjacency(mode):
    if mode not in MODES:
        raise Exception("invalid mode provided")
    
    metr_la = load_metr_la()

    if mode == "identity":
        return np.identity(len(metr_la))
    
    metr_la_adj = load_metr_la_adj()

    if mode == "rbf":
        return metr_la_adj
    
    traffic, gdf_nodes, gdf_edges = load_osm(metr_la)
    nearest_node = find_matching(metr_la, traffic)

    Q, clusters = quotient_graph(traffic, nearest_node, gdf_nodes)
    Q = relabel_graph(Q, nearest_node, clusters)

    Q1, Q2 = generate_graphs(Q, nearest_node, clusters, gdf_nodes, gdf_edges)

    adj = nx.adjacency_matrix(Q, nodelist=list(metr_la.keys())).todense()

    if mode == "osm":
        pass
    elif mode == "rbf-osm":
        adj = metr_la_adj * adj
    elif mode == "osm-length":
        matrix = generate_length_matrix(metr_la, adj, gdf_edges, nearest_node, traffic)
        adj = matrix
    elif mode == "osm-length-speed":
        matrix = generate_length_matrix(metr_la, adj, gdf_edges, nearest_node, traffic, speed=True)
        adj = matrix

    return adj, gdf_nodes, gdf_edges, Q, clusters, nearest_node, Q1, Q2

"""
Define a subgraph. Take a 40 node deterministic BFS
"""
def subgraph(G, root):
    edges = nx.bfs_edges(G, root)
    nodes = ([root] + [v for u, v in edges])[:40]
    return G.subgraph(nodes)

if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='rbf-osm')
    args = parser.parse_args()

    adj = generate_adjacency(args.mode)[0]

    with open('adj_METR-LA.pkl', 'rb') as f:
        sensor_ids, sensor_id_to_ind, _ = pickle.load(f, encoding='latin1')
        with open(f'adj-{args.mode}.pkl', 'wb') as g:
            pickle.dump((sensor_ids, sensor_id_to_ind, adj.astype(np.float32)), g)

    filename = f"adj-{args.mode}.png"
    plt.matshow(adj)
    plt.savefig(filename)

    end = time.time()
    print("completed generation of adjacency matrix for", args.mode, "in", end - start)

    # relabel to range
    metr_la = {idx: v for idx, v in enumerate(load_metr_la().values())}

    fig = plt.figure()
    plt.axis('off')
    plt.tight_layout()

    G = nx.convert_matrix.from_numpy_array(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    for edge in G.edges(data="weight"):
        nx.draw_networkx_edges(G, metr_la, edgelist=[edge], alpha = edge[2])

    nx.draw_networkx_nodes(G, metr_la, node_size=5, node_color='k')
    fig.savefig(f"graph-{args.mode}.png")