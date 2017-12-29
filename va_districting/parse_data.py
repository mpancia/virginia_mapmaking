import numpy as np
import pandas as pd
import geopandas as gpd
import igraph
from rtree import index
import os

# Locations
geojson_location = "../data/vaprecincts2013.geojson"
adj_list_location = "../data/adjacency_list.csv"
demo_location = "../data/demo_data.csv"
geo_out_location = "../data/demo_shapefile.geojson"
graph_location = "../data/graph.graphml"

if __name__ == "__main__":

    # Read data
    geo_df = gpd.read_file(geojson_location).to_crs({'init' : 'epsg:3687'})
    geo_df = geo_df.set_index(geo_df["CODE"])
    demo_df = pd.read_csv(demo_location)

    # Inspect data
    # import IPython
    # IPython.core.page.page("\n".join(demo_df.columns))

    # Pad code to correct length and set it to the index
    demo_df["CODE"] = demo_df["CODE"].astype(str).map(lambda x: x.zfill(7))
    demo_df = demo_df.set_index(demo_df["CODE"]).rename(columns={
        'Population' : 'population'
    })

    # Select wanted columns
    DEMO_COLUMNS = [
        "population"
    ]
    demo_df = demo_df[DEMO_COLUMNS]

    # Join with geo data
    joined = geo_df.join(demo_df)

    # Write joined data to file
    try:
        os.remove(geo_out_location)
    except OSError:
        pass
    joined.to_file(geo_out_location, driver='GeoJSON')

    # Interpolate with average values
    for column in DEMO_COLUMNS:
        avg_value = np.mean(joined[column])
        joined[column] = joined[column].fillna(avg_value)

    idx = index.Index()
    geo_buffer = geo_df.assign(geometry = geo_df.buffer(0))
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
        os.remove(adj_list_location)
    except OSError:
        pass

    pd.DataFrame(edges, columns=["source_id", "target_id"]).to_csv(adj_list_location, index=False)

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
    graph.write(graph_location, format='graphml')
