"""
docstring
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import igraph
import random
import attr
from collections import namedtuple
geojson_location = "../data/vaprecincts2013.geojson"
adj_list_location = "../data/adjacency_list.csv"
demo_location = "../data/demo_data.csv"

@attr.s
class Cluster(object):
    members = attr.ib()
    total_population =attr.ib()


def calculate_population_bounds(total_population, deviance_percentage, num_partitions):
    lower_bound = total_population / (num_partitions + num_partitions * deviance_percentage - deviance_percentage)
    upper_bound = total_population - (num_partitions - 1) * lower_bound
    return (int(lower_bound), int(upper_bound))

def get_total_population(cluster):
    populations = [vertex['population'] for vertex in cluster.members]
    return np.sum(populations)


def get_used_vertices(cluster_list):
    return [vertex for cluster in cluster_list for vertex in cluster.members]

def merge_clusters(cluster_list):
    """TODO: Docstring for merge_clusters.
    :returns: TODO

    """
    all_members = list(set([vertex for cluster in cluster_list for vertex in cluster.members]))
    total_population = np.sum([vertex['population'] for vertex in all_members])
    return Cluster(all_members, total_population)


def add_vertex_to_cluster(vertex, cluster):
        new_cluster_members = cluster.members + [vertex]
        new_total_population = cluster.total_population + vertex['population']
        return Cluster(new_cluster_members, new_total_population)

def add_neighbor(cluster, used_vertices):
    restricted_vertices = cluster.members + used_vertices
    all_neighbors = [neighbor for vertex in cluster.members for neighbor in vertex.neighbors() if neighbor not in restricted_vertices]
    if len(all_neighbors) > 0:
        random.shuffle(all_neighbors)
        new_member = all_neighbors.pop()
        return add_vertex_to_cluster(new_member, cluster)
    else:
        raise Exception("Cannot grow cluster!")

def grow_cluster(cluster, num_to_grow, used_vertices):
    for i in range(num_to_grow):
        cluster = add_neighbor(cluster, used_vertices)
    return cluster

def grow_cluster_to_population(cluster, min_size, used_vertices):
    while cluster.total_population < min_size:
        try:
            cluster = add_neighbor(cluster, used_vertices)
        except:
            cluster = add_neighbor(cluster, [])
    return cluster

def grow_clusters_to_min_size(cluster_list, min_size):
    min_cluster_size = min([cluster.total_population for cluster in cluster_list])
    while min_cluster_size < min_size:
        cluster_populations = [cluster.total_population for cluster in cluster_list]
        smallest_cluster_index = cluster_populations.index(min(cluster_populations))
        smallest_cluster = cluster_list[smallest_cluster_index]
        used_vertices = [vertex for cluster in cluster_list for vertex in cluster.members]
        new_cluster = grow_cluster_to_population(smallest_cluster, min_size, used_vertices)
        cluster_list[smallest_cluster_index] = new_cluster
        min_cluster_size = min([cluster.total_population for cluster in cluster_list])
    return cluster_list

def cluster_dictionary(cluster_list):
    return {vertex['name'] : cluster_id for cluster_id, cluster in enumerate(cluster_list) for vertex in cluster.members}

def label_geo_data(geo_data, cluster_list):
    geo_data = geo_data.set_index(geo_data["CODE"].astype(int).astype(str))
    label_df = pd.DataFrame.from_dict(cluster_dictionary(cluster_list), orient='index')
    label_df.columns = ['cluster']
    return geo_data.join(label_df)

def get_cluster_neigbhors(cluster):
    return set([neighbor for vertex in cluster.members for neighbor in vertex.neighbors() if neighbor not in cluster.members])

def add_vertex_to_random_cluster(vertex, cluster_list):
    valid_clusters = [(i, cluster) for i, cluster in enumerate(cluster_list) if vertex in get_cluster_neigbhors(cluster)]
    random.shuffle(valid_clusters)
    i, chosen_cluster = valid_clusters.pop()
    chosen_cluster = add_vertex_to_cluster(vertex, chosen_cluster)
    cluster_list[i] = chosen_cluster
    return cluster_list

# Load data
geo_df = gpd.read_file(geojson_location).to_crs({'init' : 'epsg:3687'})
demo_df = pd.read_csv(demo_location)
demo_df = demo_df.set_index(demo_df["CODE"].astype(int).astype(str))
adj_df = pd.read_csv(adj_list_location, dtype=int)

distinct_precincts = list(geo_df["CODE"].map(lambda x: str(int(x))))
graph = igraph.Graph()
graph.add_vertices(distinct_precincts)
adj_list =  [list(row[1].astype(str)) for row in adj_df.iterrows()]
graph.add_edges(adj_list)

# Fill in populations as node attributes
for row in demo_df[["CODE", "Population"]].itertuples():
    vertex_id = str(int(row[1]))
    population = row[2]
    if str(int(vertex_id)) in distinct_precincts:
        graph.vs.find(vertex_id)["population"] = int(np.nan_to_num(population))

# Interpolate for remaining
avg_population = np.mean(demo_df["Population"])
for vertex in list(graph.vs):
    if not vertex['population']:
        vertex['population'] = avg_population

total_population = np.sum(list(graph.vs['population']))
num_partitions = 100
deviance_percentage = 0.5

def generate_clusters(graph):
    min_size, max_size = calculate_population_bounds(total_population, deviance_percentage, num_partitions)
    cluster_vertices = random.sample(list(graph.vs), 100)
    cluster_populations = [vertex['population'] for vertex in cluster_vertices]
    clusters = [Cluster([vertex], population) for vertex, population in zip(cluster_vertices, cluster_populations)]
    # try:
    #     clusters = grow_clusters_to_min_size(clusters, min_size)
    # except:
    #     return generate_clusters(graph)
    # diff = 1
    # while diff > 0:
    #     total_size = np.sum([cluster.total_population for cluster in clusters])
    #     min_size = min_size + (total_population - total_size) * (.5 / 100)
    #     new_clusters = grow_clusters_to_min_size(clusters, min_size)
    #     if clusters == new_clusters:
    #         diff = 0
    #     else:
    #         clusters = new_clusters

    def grow_unused_vertices(clusters):
        used_vertices = get_used_vertices(clusters)
        remaining_vertices = list(set(graph.vs).difference(used_vertices))
        bad_vertices = []
        for vertex in remaining_vertices:
            try:
                clusters = add_vertex_to_random_cluster(vertex, clusters)
            except:
                bad_vertices.append(vertex)
        return bad_vertices, clusters

    bad_vertices = []
    diff = 1
    while diff > 0:
        new_bad_vertices, clusters = grow_unused_vertices(clusters)
        if set(new_bad_vertices) == set(bad_vertices):
            diff = 0
        else:
            bad_vertices = new_bad_vertices
    return clusters, bad_vertices

clusters, bad_vertices = generate_clusters(graph)
joined_geo_df = label_geo_data(geo_df, clusters)
joined_geo_df.to_file("~/Desktop/test")
