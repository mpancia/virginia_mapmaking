import pandas as pd
import geopandas as gpd
import igraph

def read_graph(adj_list_location, geojson_location):
    """Read in graph from adjacency list and GeoJSON of precincts.
    :returns: TODO

    """
    adj_df = pd.read_csv(adj_list_location, dtype=str)
    adj_list =  [list(row[1]) for row in adj_df.iterrows()]
    geo_df = gpd.read_file(geojson_location)
    distinct_precincts = list(geo_df["NAME"].unique())
    graph = igraph.Graph()
    graph.add_vertices(distinct_precincts)
    graph.add_edges(adj_list)
    return graph
