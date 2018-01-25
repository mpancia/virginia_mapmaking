import os
import random
import logging

import numpy as np

from clustering import Cluster, Clustering


class NeighborSampling(object):
    def __init__(self, initial_clustering: Clustering, geo_data):
        self.initial_clustering = initial_clustering
        logging.info("Initial clustering has {} population variance and {} political entropy." \
                     .format(initial_clustering.population_variance, \
                             initial_clustering.political_entropy))
        self.sampled_neighbors = [initial_clustering]
        self.β = 1
        self.best_variance_districts = [initial_clustering]
        self.best_comp_districts = [initial_clustering]
        self.best_racial_districts = [initial_clustering]
        self.min_variance = initial_clustering.population_variance
        self.most_comp_districts = initial_clustering.num_competitive_districts()
        self.lowest_racial_dissimilarity = initial_clustering.racial_dissimilarity
        self.geo_data = geo_data

    def calculate_energy(self, clustering: Clustering, β: float):
        political_energy = 1 / clustering.political_entropy
        population_energy = clustering.population_variance
        racial_energy = clustering.racial_dissimilarity
        total_energy = np.exp(-β * (political_energy + population_energy + racial_energy))
        return total_energy

    @staticmethod
    def find_neighboring_cluster(clustering: Clustering):
        random_neighbor, nb_clusters = random.choice(list(clustering.cluster_neighbors.items()))
        new_cluster_id = random.choice(list(nb_clusters))
        member_cluster_id = clustering.find_cluster_for_vertex(random_neighbor)
        new_clustering = clustering.copy()
        new_clustering.add_vertex_to_cluster_by_id(new_cluster_id, random_neighbor)
        if member_cluster_id:
            new_clustering.remove_vertex_from_cluster_by_id(member_cluster_id, random_neighbor)
        return new_clustering

    def find_acceptable_neighbor(self):
        acceptable = False
        most_recent_neighbor = self.sampled_neighbors[-1]
        most_recent_energy = self.calculate_energy(most_recent_neighbor, self.β)
        while not acceptable:
            candidate = self.find_neighboring_cluster(most_recent_neighbor)
            candidate_energy = self.calculate_energy(candidate, self.β)
            is_disconnected = sum([x > 1 for x in candidate.cluster_component_counts.values()]) > 0
            acceptance_ratio = np.nan_to_num(candidate_energy / most_recent_energy)
            if is_disconnected:
                acceptance_ratio = .10
                # logging.info("Rejected disconnected sample.")
                # most_recent_neighbor = random.choice(self.sampled_neighbors[::-1][:10])
            if sum([x < 1 for x in candidate.cluster_sizes.values()]) > 0:
                candidate_energy = 0
                logging.info("Rejected low population sample.")
            accept_prob = random.random()
            if acceptance_ratio >= accept_prob:
                acceptable = True
                self.sampled_neighbors = self.sampled_neighbors[::-1][:1000][::-1]
                self.sampled_neighbors.append(candidate)
                population_variance = candidate.population_variance
                competitive_districts = candidate.num_competitive_districts()
                dissimilarity = candidate.racial_dissimilarity
                if (competitive_districts >= self.most_comp_districts) and (not is_disconnected):
                    self.most_comp_districts = competitive_districts
                    self.best_comp_districts.append(candidate)
                    self.best_comp_districts = self.best_comp_districts[::-1][:10][::-1]
                if (dissimilarity <= self.lowest_racial_dissimilarity) and (not is_disconnected):
                    self.lowest_racial_dissimilarity = dissimilarity
                    self.best_racial_districts.append(candidate)
                    self.best_racial_districts = self.best_racial_districts[::-1][:10][::-1]
                if (population_variance <= self.min_variance) and (not is_disconnected):
                    self.min_variance = population_variance
                    self.best_variance_districts.append(candidate)
                    self.best_variance_districts = self.best_variance_districts[::-1][:10][::-1]
                logging.info(
                    "Accepted sample with energy {:.2e}, acceptance ratio {:.2e}, population variance {:.2e}, comp districts {}, dissimilarity index {}" \
                    .format(candidate_energy, acceptance_ratio, population_variance, competitive_districts,
                            dissimilarity))
            else:
                logging.info("Rejected sample with energy {:.2e} and acceptance ratio {:.2e}".format(candidate_energy,
                                                                                                     acceptance_ratio))


if __name__ == '__main__':
    import igraph
    from tqdm import tqdm
    import numpy as np
    from pathlib import Path
    from parse_data import (DEMO_SHAPEFILE_LOCATION, GRAPH_LOCATION,
                            POLITICAL_COMPETITION_COLUMNS)
    import geopandas as gpd

    #    logging.basicConfig(level=logging.INFO)
    LOAD_LOCATION = str(Path('../seed_maps/').resolve())
    SAVE_LOCATION = str(Path('../output_maps/').resolve())
    geo_data = gpd.read_file(DEMO_SHAPEFILE_LOCATION).set_index('CODE')
    stubs = set([os.path.splitext(x)[0] for x in os.listdir(LOAD_LOCATION)])
    initial_clusterings = [Clustering.load(os.path.join(LOAD_LOCATION, stub)) for stub in stubs]
    for clustering in initial_clusterings[::-1]:
        sampling = NeighborSampling(clustering, geo_data)
        for i in tqdm(range(10000)):
            sampling.find_acceptable_neighbor()
            if i % 1000 == 0:
                logging.info("Lowering temperature.")
                sampling.β = sampling.β * (1.1)
        for new_clustering in set(sampling.best_comp_districts + \
                                  sampling.best_variance_districts + \
                                  sampling.best_racial_districts):
            new_clustering.save(SAVE_LOCATION)
            new_clustering.save_shapefile(geo_data, SAVE_LOCATION)
