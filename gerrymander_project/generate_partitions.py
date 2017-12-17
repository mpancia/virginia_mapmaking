from sympy.utilities.iterables import multiset_partitions

def generate_possible_partitions(graph, num_partitions):
    """Generate possible partitions of a graph into a fixed number of partitions.
    :returns: TODO

    """
    
    vertex_indices = [vertex.index for vertex in graph.vs]
    possible_partition_indices = multiset_partitions(vertex_indices, num_partitions)
    possible_partitions = map(lambda partition: [graph.subgraph(subgraph_ids) for subgraph_ids in partition], possible_partition_indices)

def is_valid_partition(partition, checks):
    """Check to see if a partition satisfies a collection of checking functions.
    :returns: TODO

    """
    invalid = any(not check(partition) for check in checks)
    return not invalid

def are_subgraphs_connected(partition):
    """Check to see if the partition has all subgraphs connected. This enforces contiguity.
    :returns: TODO
    """
    is_disconnected = any(len(subgraph.components()) > 1 for subgraph in partition)
    return not is_disconnected 
