"""Classes for generating maps as clusters in precinct adjacency graph."""
import os
import uuid
import pickle
import csv
import random
import logging
import itertools

from typing import Dict, Set
from collections import defaultdict

import igraph
import numpy as np
import pandas as pd

from igraph import Graph, Vertex
from scipy.stats import entropy

from parse_data import (GRAPH_LOCATION,
                        POLITICAL_COMPETITION_COLUMNS)

# pylint: disable=E1136,E1137,E1101,C0111,R0904

class Cluster(object):
    """A Class to represent clusters."""
    def __init__(self, graph: Graph, members: Set[Vertex]):
        self._members = set(members)
        self.graph = graph
        self._neighbors = None
        self._total_population = None
        self._num_components = None
        self._political_entropy = None
        self._competitiveness = None
        self._racial_difference = None
        self._needs_update = {'total_population' : True,
                              'neighbors' : True,
                              'num_components' : True,
                              'political_entropy': True,
                              'competitiveness': True,
                              'racial_difference' : True}

    @property
    def members(self):
        return self._members

    @members.setter
    def members(self, value):
        self._members = set(value)

    @property
    def neighbors(self):
        if self._needs_update['neighbors']:
            self._neighbors = set([neighbor for member in self.members for \
                                        neighbor in member.neighbors() \
                                        if neighbor not in self.members])
            self._needs_update['neighbors'] = False
        return self._neighbors

    @property
    def num_components(self):
        if self._needs_update['num_components']:
            subgraph = self.as_subgraph()
            self._num_components = len(subgraph.components())
            self._needs_update['num_components'] = False
        return self._num_components

    @property
    def total_population(self):
        if self._needs_update['total_population']:
            self._total_population = np.sum([vertex['population'] for\
                                             vertex in self.members])
            self._needs_update['total_population'] = False
        return self._total_population

    @property
    def political_entropy(self):
        if self._needs_update['political_entropy']:
            party_pop_dict = defaultdict(float)
            party_columns = [col for col in POLITICAL_COMPETITION_COLUMNS if col != 'surveyed_total']
            party_columns = [col for col in party_columns if col != 'surveyed_blank_percentage']
            for party_column in party_columns:
                for vertex in self.members:
                    party_pop_dict[party_column] += vertex[party_column] * vertex['population']
            party_counts = np.array(list(party_pop_dict.values()))
            self._political_competition = entropy(party_counts)
            self._needs_update['political_entropy'] = False
        return self._political_competition

    @property
    def competitiveness(self):
        if self._needs_update['competitiveness']:
            total_republicans = np.sum([vertex['surveyed_republican_percentage']*vertex['population'] for vertex in self.members])
            total_democrats = np.sum([vertex['surveyed_democrat_percentage']*vertex['population'] for vertex in self.members])
            self._competitiveness = np.min([total_democrats, total_republicans]) / np.max([total_democrats, total_republicans])
            self._needs_update['competitiveness'] = False
        return self._competitiveness


    @property
    def racial_difference(self):
        if self._needs_update['racial_difference']:
            total_white = np.sum([vertex['demo_white_percentage']*vertex['population'] for vertex in self.members])
            total_black = np.sum([vertex['demo_black_percentage']*vertex['population'] for vertex in self.members])
            self._racial_difference = {'total_white' : total_white, 'total_black' : total_black}
            self._needs_update['racial_difference'] = False
        return self._racial_difference

    def add_vertex(self, vertex):
        if vertex not in self.members:
            self.members = self.members.union(set([vertex]))
            self._needs_update['total_population'] = True
            self._needs_update['neighbors'] = True
            self._needs_update['num_components'] = True
        return self

    def remove_vertex(self, vertex):
        if vertex in self.members:
            self.members = self.members.difference(set([vertex]))
            self._needs_update['total_population'] = True
            self._needs_update['neighbors'] = True
            self._needs_update['num_components'] = True
        return self

    def check_membership(self, vertex):
        return vertex in self.members

    def check_neighbor(self, vertex):
        return vertex in self.neighbors

    def check_disconnected_by_removal(self, vertex):
        if vertex in self.members:
            removed_vertices = [member for member in self.members if member != vertex]
            new_subgraph = self.graph.subgraph(removed_vertices)
            return len(new_subgraph.components()) > 1
        else:
            return False

    def check_connected(self):
        return self.num_components == 1

    def as_subgraph(self):
        return self.graph.subgraph(self.members)


class Clustering(object):

    def __init__(self, graph: Graph, clusters: Dict[int, Cluster]):
        self.graph = graph
        self.clusters = clusters
        self._unused_vertices = None
        self._used_vertices = None
        self._cluster_neighbors = None
        self._unused_cluster_neighbors = None
        self._cluster_lookup = None
        self._cluster_component_counts = None
        self._cluster_sizes = None
        self._population_variance = None
        self._population_energy = None
        self._political_entropy = None
        self._racial_dissimilarity = None
        self._needs_update = {
            'unused_vertices': True,
            'used_vertices': True,
            'cluster_neighbors': True,
            'unused_cluster_neighbors': True,
            'cluster_lookup': True,
            'cluster_component_counts': True,
            'cluster_sizes': True,
            'population_variance': True,
            'political_entropy': True,
            'racial_dissimilarity': True
        }
        self.stub = str(uuid.uuid4())

    @property
    def unused_vertices(self):
        if self._needs_update['unused_vertices']:
            all_vertices = set(list(self.graph.vs))
            used_vertices = self.used_vertices
            self._unused_vertices = all_vertices.difference(used_vertices)
            self._needs_update['unused_vertices'] = False
        return self._unused_vertices

    @property
    def used_vertices(self):
        if self._needs_update['used_vertices']:
            self._used_vertices = set.union(*[cluster.members for cluster in list(self.clusters.values())])
            self._needs_update['used_vertices'] = False
        return self._used_vertices

    @property
    def cluster_neighbors(self):
        if self._needs_update['cluster_neighbors']:
            vertex_dicts = [{neighbor : cluster_id for neighbor in list(cluster.neighbors)} for cluster_id, cluster in list(self.clusters.items())]
            keys = {k for d in vertex_dicts for k in d}
            merged_dict = {k: set([d.get(k, None) for d in vertex_dicts if d.get(k, None) is not None]) for k in keys}
            self._cluster_neighbors = merged_dict
            self._needs_update['cluster_neighbors'] = False
        return self._cluster_neighbors

    @property
    def unused_cluster_neighbors(self):
        if self._needs_update['unused_cluster_neighbors']:
            self._unused_cluster_neighbors = self.unused_vertices.intersection(self.cluster_neighbors)
            self._needs_update['unused_cluster_neighbors'] = False
        return self._unused_cluster_neighbors


    @property
    def cluster_lookup(self):
        if self._needs_update['cluster_lookup']:
            self._cluster_lookup = {vertex['name'] : cluster_id for \
                                    cluster_id, cluster in self.clusters.items() \
                                    for vertex in cluster.members}
            self._needs_update['cluster_lookup'] = False
        return self._cluster_lookup

    @property
    def cluster_component_counts(self):
        if self._needs_update['cluster_component_counts']:
            self._cluster_component_counts = {id: cluster.num_components for
                    id, cluster in list(self.clusters.items())}
            self._needs_update['cluster_component_counts'] = False
        return self._cluster_component_counts

    @property
    def cluster_sizes(self):
        if self._needs_update['cluster_sizes']:
            self._cluster_sizes = {id: cluster.total_population for \
                                   id, cluster in list(self.clusters.items())}
            self._needs_update['cluster_sizes'] = False
        return self._cluster_sizes

    @property
    def population_variance(self):
        if self._needs_update['population_variance']:
            population_sizes = np.array(list(self.cluster_sizes.values()))
            avg_population_size = np.mean(population_sizes)
            pop_variations = population_sizes/avg_population_size - 1
            self._population_energy = np.std(pop_variations)
            self._population_variance = np.max(np.abs(pop_variations))
            self._needs_update['population_variance'] = False
        return self._population_variance

    @property
    def population_energy(self):
        population_variance = self.population_variance
        return self._population_energy

    @property
    def racial_dissimilarity(self):
        if self._needs_update['racial_dissimilarity']:
            cluster_differences = [cluster.racial_difference for cluster in self.clusters.values()]
            total_black_pop = np.sum([difference['total_black'] for difference in cluster_differences])
            total_white_pop = np.sum([difference['total_white'] for difference in cluster_differences])
            cluster_dissims = [difference['total_white'] / total_white_pop - difference['total_black'] / total_black_pop \
                               for difference in cluster_differences]
            self._racial_dissimilarity = np.sum(np.abs(cluster_dissims)) * 0.5
            self._needs_update['racial_dissimilarity'] = False
        return self._racial_dissimilarity

    @property
    def political_entropy(self):
        if self._needs_update['political_entropy']:
            cluster_entropies = [cluster.political_entropy for cluster in self.clusters.values()]
            self._political_entropy = np.std([comp for comp in cluster_entropies if np.isfinite(comp)])
            self._needs_update['political_entropy'] = False
        return self._political_entropy

    def num_competitive_districts(self):
        competitive_districts = [x for x in self.clusters.values() if x.competitiveness > .8]
        return len(competitive_districts)

    def add_vertex_to_cluster_by_id(self, cluster_id: int, vertex: Vertex):
        self.clusters[cluster_id] = self.clusters[cluster_id].add_vertex(vertex)
        self._needs_update = dict.fromkeys(self._needs_update, True)
        return self

    def remove_vertex_from_cluster_by_id(self, cluster_id: int, vertex: Vertex):
        self.clusters[cluster_id] = self.clusters[cluster_id].remove_vertex(vertex)
        self._needs_update = dict.fromkeys(self._needs_update, True)
        return self

    def get_cluster_neighbors_by_id(self, cluster_id: int):
        return self.clusters[cluster_id].neighbors


    def label_geo_data(self, geo_data):
        cluster_df = pd.DataFrame.from_dict(self.cluster_lookup, orient='index')
        cluster_df.columns = ['cluster']
        return geo_data.join(cluster_df)

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

        return None

    def add_random_neighbor_to_cluster(self, cluster_id: int):
        neighbors = self.get_cluster_neighbors_by_id(cluster_id)
        unused_vertices = self.unused_vertices
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

    def get_smallest_cluster_id(self):
        cluster_populations = list(self.cluster_sizes.items())
        cluster_populations.sort(key=lambda x: - x[1])
        smallest_cluster_id = cluster_populations.pop()[0]
        return smallest_cluster_id

    def grow_smallest_cluster(self):
        smallest_cluster_id = self.get_smallest_cluster_id()
        return self.add_random_neighbor_to_cluster(smallest_cluster_id)

    def grow_to_min_size(self, min_size):
        while min([x for x in list(self.cluster_sizes.values())]) < min_size:
            self.grow_smallest_cluster()
        return self

    def add_unused_neighbor(self):
        vertex = random.choice(list(self.unused_cluster_neighbors))
        vertex_neighbors = self.find_neighbor_for_vertex(vertex)
        cluster_to_add_to = random.choice(vertex_neighbors)
        return self.add_vertex_to_cluster_by_id(cluster_to_add_to, vertex)

    def save(self, location):
        graph_path = os.path.join(location, self.stub) + '.graph'
        self.graph.write_picklez(graph_path)
        lookup_path = os.path.join(location, self.stub) + '.clusters'
        with open(lookup_path, 'wb') as f:
            pickle.dump(self.cluster_lookup, f)
        return self

    def copy(self):
        cluster_lookup = [(self.graph.vs.find(id), cluster_id) for id, cluster_id in self.cluster_lookup.items()]
        cluster_groups = itertools.groupby(cluster_lookup, lambda x: x[1])
        clusters = {id : Cluster(graph=self.graph, members=set([x[0] for x in members])) for id, members in cluster_groups}
        return Clustering(graph=self.graph, clusters=clusters)

    def save_shapefile(self, geo_data, save_location):
        manifest_path = os.path.join(save_location, 'manifest.csv')
        if not os.path.exists(manifest_path):
            row = ['path', 'variance', 'disconnected_components', 'competitive_districts', 'racial_dissimilarity']
            with open(manifest_path, "w") as file:
                writer = csv.writer(file)
                writer.writerow(row)

        logging.info('Written to {}'.format(save_location))
        disconnected_count = np.sum([x for x in self.cluster_component_counts.values() if x > 1])
        row = [self.stub + '.geojson', self.population_variance, disconnected_count, self.num_competitive_districts(), self.racial_dissimilarity]
        with open(manifest_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(row)
        shp_location = os.path.join(save_location, self.stub + '.geojson')
        try:
            os.remove(shp_location)
        except OSError:
            pass
        labeled_df = self.label_geo_data(geo_data)
        labeled_df = labeled_df.fillna({'cluster' : 99})
        labeled_df['geometry'] = labeled_df.buffer(0)
        labeled_df.to_file(shp_location, driver="GeoJSON")



    @classmethod
    def from_random_vertices(cls, seed_graph: Graph, num_clusters: int):
        """ Generate a clustering by picking a random set of vertices.
        """
        random_vertices = random.sample(list(seed_graph.vs), num_clusters)
        clusters = [Cluster(graph=seed_graph, members=set([vertex])) for vertex in random_vertices]
        return cls(seed_graph, {i : cluster for i, cluster in enumerate(clusters)})

    @classmethod
    def load(cls, location):
        graph_path = location + '.graph'
        graph = Graph.Read_Picklez(graph_path)
        lookup_path = location + '.clusters'
        with open(lookup_path, 'rb') as f:
            id_cluster_lookup = pickle.load(f)
        cluster_lookup = [(graph.vs.find(id), cluster_id) for id, cluster_id in id_cluster_lookup.items()]
        cluster_groups = itertools.groupby(cluster_lookup, lambda x: x[1])
        clusters = {id : Cluster(graph=graph, members=set([x[0] for x in members])) for id, members in cluster_groups}
        return cls(graph=graph, clusters=clusters)

if __name__ == "__main__":
    import csv

    from pathlib import Path

    NUM_TO_GENERATE = 10
    logging.basicConfig(level=logging.INFO)

    graph = igraph.read(GRAPH_LOCATION, format='graphml')
    large_component = graph.subgraph(graph.components()[0])

    def generate_map(component):
        clustering = Clustering.from_random_vertices(component, 99)
        try:
            clustering = clustering.grow_to_min_size(30000)
        except:
            logging.error('Generating min size failed.')
            return generate_map(component)

        unused_vertices = len(clustering.unused_vertices)
        while unused_vertices > 0:
            try:
                clustering = clustering.add_unused_neighbor()
                unused_vertices = len(clustering.unused_vertices)
            except:
                logging.error('Remainder allocation failed with {} unused vertices.'.format(unused_vertices))
                return generate_map(component)

        unused_vertices = len(clustering.unused_vertices)
        if unused_vertices > 0:
            logging.error('Remainder allocation failed with {} unused vertices.'.format(unused_vertices))
            return generate_map(component)

        disconnected_count = np.sum([x for x in clustering.cluster_component_counts.values() if x > 1])
        if disconnected_count > 0:
            logging.error('{} disconnected components.'.format(disconnected_count))
            return generate_map(component)

        if clustering.population_variance > 5:
            logging.error('Pop variance too high: {}'.format(clustering.population_variance))
            return generate_map(component)
        else:
            return clustering

    for i in range(NUM_TO_GENERATE):
        clustering = generate_map(large_component)
        logging.info('Generated map with {} population variance.'.format(clustering.population_variance))
        clustering.save(str(Path('../seed_maps').resolve()))
