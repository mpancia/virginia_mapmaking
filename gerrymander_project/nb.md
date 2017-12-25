```python
adj_list_location = "../data/adjacency_list.csv"
geojson_location = "../data/vaprecincts2013.geojson"
```

```python
import pandas as pd
import geopandas as gpd
import igraph
from rtree import index
import random
```

```python
geo_df = gpd.read_file(geojson_location).to_crs({'init' : 'epsg:3687'})
```

```python
geo_df[geo_df["PRECINCT"] == '001']
```

```python
idx = index.Index()
```

```python
geo_buffer = geo_df.assign(geometry = geo_df.buffer(50))
```

```python
geo_buffer.plot()
```

```python
for i, poly in geo_buffer.iterrows():
    idx.insert(i, poly.geometry.bounds)
```

```python
geo_buffer_list = [x[1] for x in list(geo_buffer.iterrows())]
precinct = geo_buffer_list[10]
geo_df.iloc[list(idx.intersection(precinct.geometry.bounds))].plot()
```

```python
precinct = [x for x in geo_buffer_list if x.CODE == '1310401'][0]
geo_df.iloc[list(idx.intersection(precinct.geometry.bounds))].plot()

```

```python
edges = []
for i, precinct in geo_df.iterrows():
    precinct_id = precinct.CODE
    matching_precincts = list(idx.intersection(precinct.geometry.bounds))
    matching_precinct_ids = list(geo_df.iloc[matching_precincts].CODE)
    new_edges = [(precinct_id, matched_id) for matched_id in matching_precinct_ids]
    edges += new_edges
```

```python
len(edges)
```

```python
pd.DataFrame(edges, columns=["source_id", "target_id"]).to_csv(adj_list_location, index=False)
```

```python
adj_df = pd.read_csv(adj_list_location, dtype=str)
adj_list =  [list(row[1]) for row in adj_df.iterrows()]
distinct_precincts = list(geo_df["CODE"].unique())
graph = igraph.Graph()
graph.add_vertices(distinct_precincts)
graph.add_edges(adj_list)
```

```python
len(graph.vs)
```

```python
len(graph.es)
```

```python
len(geo_df)
```

```python
len(graph.components())
```

```python
node = graph.vs[0]

```

```python
def get_subgraph_nbhd(subgraph, graph):
    subgraph_indices = [vertex['name'] for vertex in subgraph.vs]
    nbhd_indices = [graph.vs[x]['name'] for y in graph.neighborhood(subgraph_indices) for x in y if graph.vs[x]['name'] not in subgraph_indices]
    return nbhd_indices
```

```python
subgraph = graph.subgraph([0, 100])
new_indices = get_subgraph_nbhd(subgraph, graph)
```

```python
added_vertex = random.choice(new_indices)
```

```python
new_subgraph = graph.subgraph(subgraph.vs['name'] + [added_vertex])
```

```python
list(new_subgraph.vs)
```

```python
def add_random_neighbor(subgraph, graph, used_vertices):
    new_indices = [index for index in get_subgraph_nbhd(subgraph, graph) if index not in used_vertices]
    if len(new_indices) > 0:
        added_vertex = random.choice(new_indices)
        subgraph = graph.subgraph(subgraph.vs['name'] + [added_vertex])
        return added_vertex, subgraph
    else: 
        return None, subgraph

```

```python
subgraph = graph.subgraph([1000])
for i in range(500):
    _, subgraph = add_random_neighbor(subgraph, graph)
```

```python
names = list(subgraph.vs['name'])
```

```python
geo_df[list(map(lambda x: x in names, geo_df.CODE))].plot()
```

```python
seed_vertices = random.choices(list(graph.vs['name']), k=11)
```

```python
district_ids = range(len(seed_vertices))
used_vertices = seed_vertices
unused_vertices = [name for name in list(graph.vs['name']) if name not in used_vertices]
subgraphs = [graph.subgraph(vertex) for vertex in seed_vertices]
```

```python
district_ids = range(len(seed_vertices))
while len(unused_vertices) > 22:
    random_subgraph_id = random.choice(district_ids)
    random_subgraph = subgraphs[random_subgraph_id]
    added_vertex, new_subgraph = add_random_neighbor(random_subgraph, graph, used_vertices)
    subgraphs[random_subgraph_id] = new_subgraph
    if added_vertex:
        used_vertices += [added_vertex]
        unused_vertices.remove(added_vertex)
```

```python
for id in district_ids:
    district = subgraphs[id]
    district_ids = list(district.vs['name'])
    geo_df.loc[geo_df['CODE'].isin(district_ids), "district"] = id
```

```python
geo_df.plot(column = "district")
```

```python
geo_df[geo_df["district"].isnull()]
```

```python
import scipy as sp
from scipy.sparse import lil_matrix
import numpy as np
```

```python
num_precincts = len(distinct_precincts)
index_to_precinct = dict(enumerate(distinct_precincts))
precinct_to_index = {v:k for (k,v) in index_to_precinct.items()}
```

```python
adj_mat_lil = lil_matrix((num_precincts, num_precincts), dtype=np.float32)
entries = [(precinct_to_index[x], precinct_to_index[y]) for x,y in adj_list]
for x,y in entries:
    if x != y:
        adj_mat_lil[x,y] = 1
        adj_mat_lil[y,x] = 1
```

```python
adj_mat = adj_mat_lil.tocsr()
```

```python
degrees = np.asarray(np.sum(adj_mat, axis=0)).squeeze()
```

```python
laplacian_mat_lil = lil_matrix((num_precincts, num_precincts), dtype=np.float32)
for x,y in entries:
    if x != y:
        deg_x = degrees[x]
        deg_y = degrees[y]    
        val = - 1/np.sqrt((deg_x * deg_y))
        laplacian_mat_lil[x,y] = val
        laplacian_mat_lil[y,x] = val
for x in range(num_precincts):
    laplacian_mat_lil[x,x] = 1
```

```python
laplacian_mat = laplacian_mat_lil.tocsr()
```

```python
alpha = 0.001
num_districts = 11
import random
random_starts = random.choices(range(num_precincts), k=num_districts)
label_matrix_lil = lil_matrix((num_precincts, num_districts))
for district_num, start in enumerate(random_starts):
    label_matrix_lil[start, district_num] = 1
```

```python
initial_label_matrix = label_matrix_lil.tocsr()
```

```python
label_matrix = initial_label_matrix.copy()
```

```python
for i in range(1000):
    label_matrix = alpha*np.dot(laplacian_mat, label_matrix) + (1-alpha)*initial_label_matrix
```

```python
districts = np.asarray(np.argmax(label_matrix, axis=1)).squeeze()
```

```python
for i in range(num_precincts):
    precinct_code = index_to_precinct[i]
    district_id = districts[i]
    geo_df.loc[geo_df['CODE'] == precinct_code, "district"] = district_id
```

```python
%matplotlib inline
geo_df.plot(column = "district")

```

```python
geo_df.to_file(driver='GeoJSON', filename="/Users/msp/Desktop/t.geojson")
```
