from clustering import Cluster, Clustering
import copy
import logging
import os
import random

class NeighborSampling(object):
    def __init__(self, initial_clustering: Clustering):
        self.initial_clustering = initial_clustering
        logging.info("Initial clustering has {} population variance and {} political entropy."\
                    .format(initial_clustering.population_variance, \
                            initial_clustering.political_competition))
        self.sampled_neighbors = []

    @staticmethod
    def find_neighboring_cluster(clustering: Clustering):
        random_neighbor, nb_clusters = random.choice(list(clustering.cluster_neighbors.items()))
        new_cluster_id = random.choice(list(nb_clusters))
        member_cluster_id = clustering.find_cluster_for_vertex(random_neighbor)
        new_clustering = Clustering(clusters=clustering.clusters, graph=clustering.graph)
        new_clustering.add_vertex_to_cluster_by_id(new_cluster_id, random_neighbor)
        if member_cluster_id:
            new_clustering.remove_vertex_from_cluster_by_id(member_cluster_id, random_neighbor)
        return new_clustering

if __name__ == '__main__':
    import igraph
    import numpy as np
    from parse_data import DEMO_SHAPEFILE_LOCATION, GRAPH_LOCATION, POLITICAL_COMPETITION_COLUMNS
    logging.basicConfig(level=logging.INFO)
    graph = igraph.read(GRAPH_LOCATION, format='graphml')
    large_component = graph.subgraph(graph.components()[0])
    clustering = Clustering.from_random_vertices(large_component, 2)
    sampling = NeighborSampling(clustering)
    sampling.find_neighboring_cluster(clustering)


