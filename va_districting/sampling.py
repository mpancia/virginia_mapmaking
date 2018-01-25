import logging
import os
import random

from clustering import Clustering


class NeighborSampling(object):
    """A class to represent a sampling process for generating new clusterings."""

    def __init__(self, initial_clustering: Clustering, geo_data):
        """

        Args:
            initial_clustering (Clustering): A seed clustering to start sampling around.
            geo_data (DataFrame): A shapefile of geographic data used for geographic calculations.

        Attributes:
            sampled_neighbors (list): A running list of the most recent 1000 samples.
            β (float): A temperature parameter for sampling.
            best_variance_districts (list): A running list of the best 10 sampled maps (lowest population variance).
            best_comp_districts (list): A running list of the best 10 sampled maps
                (highest num of politically competitive districts).
            best_racial_districts (list): A running list of the best 10 sampled map (lowest racial dissimilarity index).
            min_variance (float): The current minimum population variance.
            most_comp_districts (int): The current max number of competitive districts.
            lowest_racial_dissimilarity (float): The current lowest racial dissimilarity index.

        Properties:

        """
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
        """Calculate the current energy for sampling.

        The sampling process looks to sample based on probability of a sample being related to its energy.
        Low energy samples are thought of as more likely.

        This is computed as an exponential function of 3 components coming from

        1. political competitveness (higher -> lower energy),
        2. population variance (lower -> lower energy),
        3. racial dissimilarity (lower -> lower energy).

        They are each given a weight of 1, which could be tweaked to change the sampling process.

        Args:
            clustering: A proposed clustering.
            β: A temperature parameter.

        Returns:
            float: The energy.

        """
        political_energy = 1 / clustering.political_entropy
        population_energy = clustering.population_variance
        racial_energy = clustering.racial_dissimilarity
        total_energy = np.exp(-β * (political_energy + population_energy + racial_energy))
        return total_energy

    @staticmethod
    def find_neighboring_cluster(clustering: Clustering):
        """Find a nearby clustering that is obtained by changing a random vertex's membership."""
        random_neighbor, nb_clusters = random.choice(list(clustering.cluster_neighbors.items()))
        new_cluster_id = random.choice(list(nb_clusters))
        member_cluster_id = clustering.find_cluster_for_vertex(random_neighbor)
        new_clustering = clustering.copy()
        new_clustering.add_vertex_to_cluster_by_id(new_cluster_id, random_neighbor)
        if member_cluster_id:
            new_clustering.remove_vertex_from_cluster_by_id(member_cluster_id, random_neighbor)
        return new_clustering

    def find_acceptable_neighbor(self):
        """Find an acceptable nearby sample."""
        acceptable = False
        most_recent_neighbor = self.sampled_neighbors[-1]
        most_recent_energy = self.calculate_energy(most_recent_neighbor, self.β)
        while not acceptable:
            candidate = self.find_neighboring_cluster(most_recent_neighbor)
            candidate_energy = self.calculate_energy(candidate, self.β)
            is_disconnected = sum([x > 1 for x in candidate.cluster_component_counts.values()]) > 0
            # Acceptance ratio is the ratio of the new canditate energy to the most recent one in the sampling process.
            acceptance_ratio = np.nan_to_num(candidate_energy / most_recent_energy)
            if is_disconnected:
                # Allow for motion through disconnected neighbors, but only with a .1 probability.
                acceptance_ratio = .10
            if sum([x < 1 for x in candidate.cluster_sizes.values()]) > 0:
                # Reject samples that have 0-population clusters.
                candidate_energy = 0
                logging.info("Rejected low population sample.")
            accept_prob = random.random()
            if acceptance_ratio >= accept_prob:
                # Randomly accept a sample if the acceptance ratio is greater than a random number in [0,1].
                acceptable = True
                self.sampled_neighbors = self.sampled_neighbors[::-1][:1000][::-1]
                self.sampled_neighbors.append(candidate)

                # Calculate statistics and append the sample to the appropriate "best of" list
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
    from tqdm import tqdm
    import numpy as np
    from pathlib import Path
    from parse_data import (DEMO_SHAPEFILE_LOCATION)
    import geopandas as gpd

    logging.basicConfig(level=logging.INFO)
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
