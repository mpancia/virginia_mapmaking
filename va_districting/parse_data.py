"""Parse electoral data."""
import os

import igraph
import numpy as np
import pandas as pd
import geopandas as gpd

from rtree import index

# Locations
GEOJSON_LOCATION = "../data/vaprecincts2013.geojson"
ADJ_LIST_LOCATION = "../data/adjacency_list.csv"
DEMOGRAPHIC_LOCATION = "../data/demo_data.csv"
DEMO_SHAPEFILE_LOCATION = "../data/demo_shapefile.geojson"
GRAPH_LOCATION = "../data/graph.graphml"

if __name__ == "__main__":

    # Read data
    geo_df = gpd.read_file(GEOJSON_LOCATION).to_crs({'init' : 'epsg:3687'})
    geo_df = geo_df.set_index(geo_df["CODE"])
    demo_df = pd.read_csv(DEMOGRAPHIC_LOCATION)

    # Pad code to correct length and set it to the index
    demo_df["CODE"] = demo_df["CODE"].astype(str).map(lambda x: x.zfill(7))
    demo_df = (demo_df
               .set_index(demo_df["CODE"])
               .rename(columns={
                   'Population' : 'population',
                   'Democratic' : 'surveyed_democrat',
                   'Independent' : 'surveyed_independent',
                   'Independent Green' : 'surveyed_independent_green',
                   'Libertarian' : 'surveyed_libertarian',
                   'Republican' : 'surveyed_republican',
                   'Grand Total' : 'surveyed_total',
                   '(blank)' : 'surveyed_blank'})
               .assign(
                   surveyed_democrat_percentage=lambda x: 100*x['surveyed_democrat'] / x['surveyed_total'],
                   surveyed_independent_percentage=lambda x: 100*x['surveyed_independent'] / x['surveyed_total'],
                   surveyed_independent_green_percentage=lambda x: 100*x['surveyed_independent_green'] / x['surveyed_total'],
                   surveyed_libertarian_percentage=lambda x: 100*x['surveyed_libertarian'] / x['surveyed_total'],
                   surveyed_blank_percentage=lambda x: 100*x['surveyed_blank'] / x['surveyed_total'],
                   surveyed_republican_percentage=lambda x: 100*x['surveyed_republican'] / x['surveyed_total']
               ))

    # Select wanted columns
    DEMO_COLUMNS = [
        'population',
        'surveyed_democrat_percentage',
        'surveyed_independent_green_percentage',
        'surveyed_independent_percentage',
        'surveyed_republican_percentage',
        'surveyed_blank_percentage',
        'surveyed_libertarian_percentage',
        'surveyed_total'
    ]
    demo_df = demo_df[DEMO_COLUMNS]

    # Fill NaN with 0 for survey responses
    demo_df = demo_df.fillna(value={column: 0 for column in DEMO_COLUMNS if column != 'population'})

    # Join with geo data
    joined = geo_df.join(demo_df)

    # Write joined data to file
    try:
        os.remove(DEMO_SHAPEFILE_LOCATION)
    except OSError:
        pass
    joined.to_file(DEMO_SHAPEFILE_LOCATION, driver='GeoJSON')

    # Interpolate with average values
    for column in DEMO_COLUMNS:
        avg_value = np.mean(joined[column])
        joined[column] = joined[column].fillna(avg_value)

    idx = index.Index()
    geo_buffer = geo_df.assign(geometry=geo_df.buffer(0))
    for i, poly in geo_buffer.iterrows():
        idx.insert(int(i), poly.geometry.bounds)

    edges = []
    for i, precinct in geo_buffer.iterrows():
        precinct_id = precinct.CODE
        matching_precincts = list(idx.intersection(precinct.geometry.bounds))
        matching_precincts = [str(match).zfill(7) for match in matching_precincts]
        distances = geo_buffer.loc[matching_precincts].distance(precinct.geometry)
        for match_id, distance in distances.items():
            if distance < 10:
                edges += [(precinct_id, match_id)]

    try:
        os.remove(ADJ_LIST_LOCATION)
    except OSError:
        pass

    pd.DataFrame(edges, columns=["source_id", "target_id"]).to_csv(ADJ_LIST_LOCATION, index=False)

    # Create graph
    distinct_precincts = list(joined["CODE"])
    graph = igraph.Graph()
    graph.add_vertices(distinct_precincts)
    graph.add_edges(edges)

    # Add metadata to graph
    for column in DEMO_COLUMNS:
        for row in joined[["CODE", column]].itertuples():
            vertex_id = row[1]
            value = row[2]
            if vertex_id in distinct_precincts:
                graph.vs.find(vertex_id)[column] = np.nan_to_num(value)

    # Write graph
    graph.write(GRAPH_LOCATION, format='graphml')
