import random
from typing import Dict, List

import attr
import igraph
from igraph import Graph, Vertex
import pandas as pd
import numpy as np
import os
import logging
logging.basicConfig(level=logging.INFO)

# pylint: disable=E1136,E1137,E1101

geo_out_location = "../data/demo_shapefile.geojson"
graph_location = "../data/graph.graphml"


@attr.s
class Cluster(object):
    members = attr.ib(type=List[Vertex])
    graph = attr.ib(type=Graph)

    def add_vertex(self, vertex):
        self.members.append(vertex)
        self.members = list(set(self.members))
        return self

    def remove_vertex(self, vertex):
        self.members = [member for member in self.members if member != vertex]
        return self

    def get_neighbors(self):
        return list(set([neighbor for member in self.members for neighbor in member.neighbors() \
                    if neighbor not in self.members]))

    def check_membership(self, vertex):
        return vertex in self.members

    def check_neighbor(self, vertex):
        return vertex in self.get_neighbors()

    def get_total_population(self):
        return np.sum([vertex['population'] for vertex in self.members])

    def as_subgraph(self):
        return self.graph.subgraph(self.members)

    def check_disconnected_by_removal(self, vertex):
        removed_vertices = [member for member in self.members if member != vertex]
        new_subgraph = self.graph.subgraph(removed_vertices)
        return len(new_subgraph.components()) > 1

    def check_connected(self):
        subgraph = self.as_subgraph()
        return len(subgraph.components()) == 1


@attr.s
class Clustering(object):
    graph = attr.ib(type=igraph.Graph)
    clusters = attr.ib(type=Dict[int, Cluster])

    def add_vertex_to_cluster_by_id(self, cluster_id: int, vertex: Vertex):
        self.clusters[cluster_id] = self.clusters[cluster_id].add_vertex(vertex)
        return self

    def remove_vertex_from_cluster_by_id(self, cluster_id: int, vertex: Vertex):
        self.clusters[cluster_id] = self.clusters[cluster_id].remove_vertex(vertex)
        return self

    def get_cluster_neighbors_by_id(self, cluster_id: int):
        return self.clusters[cluster_id].get_neighbors()

    def get_all_cluster_neighbors(self):
        return [neighbor for cluster_id in self.clusters.keys() \
                for neighbor in self.get_cluster_neighbors_by_id(cluster_id)]

    def get_used_vertices(self):
        return set([member for cluster in list(self.clusters.values()) for member in cluster.members])

    def get_unused_vertices(self):
        used_vertices = self.get_used_vertices()
        all_vertices = set(self.graph.vs)
        return list(all_vertices.difference(used_vertices))

    def get_cluster_lookup(self):
        return {vertex['name'] : cluster_id for cluster_id, cluster in self.clusters.items() \
                for vertex in cluster.members}

    def label_geo_data(self, geo_data):
        cluster_lookup = self.get_cluster_lookup()
        cluster_df = pd.DataFrame.from_dict(cluster_lookup, orient='index')
        cluster_df.columns = ['cluster']
        return geo_data.join(cluster_df)

    def get_all_unused_cluster_neighbors(self):
        unused_vertices = set(self.get_unused_vertices())
        cluster_neighbors = set(self.get_all_cluster_neighbors())
        return list(unused_vertices.intersection(cluster_neighbors))

    def find_cluster_for_vertex(self, vertex: Vertex):
        cluster_ids = [cluster_id for cluster_id in self.clusters.keys() \
                       if self.clusters[cluster_id].check_membership(vertex)]
        if len(cluster_ids) == 1:
            return cluster_ids[0]
        else:
            return None

    def find_neighbor_for_vertex(self, vertex: Vertex):
        cluster_ids = [cluster_id for cluster_id in self.clusters.keys() \
                       if self.clusters[cluster_id].check_neighbor(vertex)]
        if len(cluster_ids) > 0:
            return cluster_ids
        else:
            return None


    def add_random_neighbor_to_cluster(self, cluster_id: int):
        neighbors = set(self.get_cluster_neighbors_by_id(cluster_id))
        unused_vertices = set(self.get_unused_vertices())
        unused_neighbors = neighbors.intersection(unused_vertices)
        if len(unused_neighbors) > 0:
            vertex_to_add = random.choice(list(unused_neighbors))
            self.add_vertex_to_cluster_by_id(cluster_id=cluster_id,
                                             vertex=vertex_to_add)
        else:
            # neighbor_degrees = []
            # for neighbor in neighbors:
            #     degree = sum([neighbor in member.neighbors() for member in self.clusters[cluster_id].members])
            #     neighbor_degrees.append((neighbor, degree))
            # neighbor_degrees.sort(key=lambda x: x[1])
            # low_deg_neighbor = neighbor_degrees.pop()[0]
            # cluster_for_neighbor = self.find_cluster_for_vertex(low_deg_neighbor)
            vertex_to_add = random.choice(list(neighbors))
            cluster_for_neighbor = self.find_cluster_for_vertex(vertex_to_add)
            self.remove_vertex_from_cluster_by_id(cluster_id=cluster_for_neighbor,
                                                  vertex=vertex_to_add)
            self.add_vertex_to_cluster_by_id(cluster_id=cluster_id,
                                             vertex=vertex_to_add)
        return self

    def get_cluster_sizes(self):
        return [(cluster_id, self.clusters[cluster_id].get_total_population()) for\
                               cluster_id in self.clusters.keys()]

    def get_smallest_cluster_id(self):
        cluster_populations = self.get_cluster_sizes()
        cluster_populations.sort(key=lambda x: - x[1])
        smallest_cluster_id = cluster_populations.pop()[0]
        return smallest_cluster_id

    def grow_smallest_cluster(self):
        smallest_cluster_id = self.get_smallest_cluster_id()
        return self.add_random_neighbor_to_cluster(smallest_cluster_id)

    def grow_to_min_size(self, min_size):
        while min([x[1] for x in self.get_cluster_sizes()]) < min_size:
            self.grow_smallest_cluster()
        return self

    def get_current_population_variance(self):
        population_sizes = [x[1] for x in self.get_cluster_sizes()]
        smallest_size = min(population_sizes)
        largest_size = max(population_sizes)
        return abs(largest_size - smallest_size) / smallest_size

    def add_unused_neighbor(self):
        unused_neighbors = self.get_all_unused_cluster_neighbors()
        vertex = random.choice(unused_neighbors)
        vertex_neighbors = self.find_neighbor_for_vertex(vertex)
        cluster_to_add_to = random.choice(vertex_neighbors)
        return self.add_vertex_to_cluster_by_id(cluster_to_add_to, vertex)

    def get_disconnected_component_count(self):
        disconnectivities = [not cluster.check_connected() \
                             for cluster in self.clusters.values()]
        return sum(disconnectivities)

    def check_connected(self):
        disconnectivities = self.get_disconnected_component_count()
        return disconnectivities == 0

    @classmethod
    def from_random_vertices(cls, seed_graph: Graph, num_clusters: int):
        """ Generate a clustering by picking a random set of vertices.
        """
        random_vertices = random.sample(list(seed_graph.vs), num_clusters)
        clusters = [Cluster([vertex], seed_graph) for vertex in random_vertices]
        return cls(seed_graph, {i : cluster for i, cluster in enumerate(clusters)})

if __name__ == "__main__":
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import uuid
    import csv
    graph = igraph.read(graph_location, format='graphml')
    large_component = graph.subgraph(graph.components()[0])
    total_large_population = np.sum(list(large_component.vs['population']))

    def generate_map(component):
        clustering = Clustering.from_random_vertices(component, 99)
        try:
            clustering = clustering.grow_to_min_size(20000)
        except:
            logging.error('Generating min size failed.')
            return generate_map(component)
        unused_vertices = len(clustering.get_unused_vertices())
        while unused_vertices > 0:
            try:
                clustering = clustering.add_unused_neighbor()
                unused_vertices = len(clustering.get_unused_vertices())
            except:
                logging.error('Remainder allocation failed with {} unused vertices.'.format(unused_vertices))
                return generate_map(component)

        unused_vertices = len(clustering.get_unused_vertices())
        if unused_vertices > 0:
            logging.error('Remainder allocation failed with {} unused vertices.'.format(unused_vertices))
            return generate_map(component)

        variance = clustering.get_current_population_variance()

        if variance > 15:
            logging.error('Pop variance too high: {}'.format(variance))
            return generate_map(component)
        else:
            return clustering, variance

    geo_df = gpd.read_file(geo_out_location).to_crs({'init' : 'epsg:3687'}).set_index('CODE')
    manifest_path = os.path.join('/Users', 'msp', 'Desktop', 'shapes', 'manifest.csv')
    row = ['path', 'variance', 'disconnected_components']
    with open(manifest_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(row)

    for i in range(10):
        clustering, variance = generate_map(large_component)
        disconnected_count = clustering.get_disconnected_component_count()
        labeled_df = clustering.label_geo_data(geo_df)
        labeled_df = labeled_df.fillna({'cluster' : 99})
        labeled_df['geometry'] = labeled_df.buffer(0)

        logging.info('Generated map with {} disconnected components and {} population variance.'.format(disconnected_count, variance))

        stub = str(uuid.uuid4())
        write_path = os.path.join('/Users', 'msp', 'Desktop', 'shapes', stub + '.geojson')
        logging.info('Written to {}'.format(write_path))
        row = [stub + '.geojson', variance, disconnected_count]
        with open(manifest_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(row)

        try:
            os.remove(write_path)
        except OSError:
            pass
        labeled_df.to_file(write_path, driver="GeoJSON")

    # dissolved_df = geo_df.dissolve(by='cluster').reset_index()
    # dissolved_df['coords'] = dissolved_df['geometry'].apply(lambda x: x.representative_point().coords[:])
    # dissolved_df['coords'] = [coords[0] for coords in dissolved_df['coords']]
    # ax = dissolved_df.plot(column='cluster')
    # for dx, row in dissolved_df.iterrows():
    #     plt.annotate(s=int(row['cluster']), xy=row['coords'], horizontalalignment='center')
