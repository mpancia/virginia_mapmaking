import os
import copy
import random
import logging

import numpy as np

from clustering import Cluster, Clustering


class NeighborSampling(object):
    def __init__(self, initial_clustering: Clustering):
        self.initial_clustering = initial_clustering
        logging.info("Initial clustering has {} population variance and {} political entropy."\
                    .format(initial_clustering.population_variance, \
                            initial_clustering.political_competition))
        self.sampled_neighbors = [initial_clustering]

    @staticmethod
    def calculate_energy(clustering):
        political_competition = clustering.political_competition
        political_energy = np.exp(political_competition) - 1
        population_variance = clustering.population_variance
        population_energy = np.exp(population_variance) - 1
        total_energy = political_energy * population_energy
        return total_energy

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

    def find_acceptable_neighbor(self):
        acceptable = False
        most_recent_neighbor = self.sampled_neighbors[-1]
        most_recent_energy = self.calculate_energy(most_recent_neighbor)
        while not acceptable:
            candidate = self.find_neighboring_cluster(most_recent_neighbor)
            candidate_energy = self.calculate_energy(candidate)
            acceptance_ratio = np.nan_to_num(candidate_energy / most_recent_energy)
            accept_prob = random.random()
            if acceptance_ratio <= accept_prob:
                acceptable = True
                self.sampled_neighbors.append(candidate)
                logging.info("Accepted sample with energy {}".format(candidate_energy))
            else:
                logging.info("Rejected sample with energy {} and acceptance ratio {}".format(candidate_energy, acceptance_ratio))


if __name__ == '__main__':
    import igraph
    import numpy as np
    from pathlib import Path
    SAVE_LOCATION = str(Path('../test2').resolve())
    from parse_data import (DEMO_SHAPEFILE_LOCATION, GRAPH_LOCATION,
                            POLITICAL_COMPETITION_COLUMNS)

    logging.basicConfig(level=logging.INFO)
    clustering = Clustering.load(SAVE_LOCATION)
    sampling = NeighborSampling(clustering)
    # for i in range(2000):
    #     sampling.find_acceptable_neighbor()


