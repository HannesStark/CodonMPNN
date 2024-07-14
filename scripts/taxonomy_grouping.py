import pandas as pd
import matplotlib.pyplot as plt
from ete3 import NCBITaxa, Tree

# read in afdb database
afdb_df = pd.read_csv("afdb_with_reassigned_tids.csv")

# reassign ids that were merged in the new NCBI dataset
afdb_df.loc[afdb_df['taxon_id'] == 47238, 'taxon_id'] = 3140254
afdb_df.loc[afdb_df['taxon_id'] == 4111, 'taxon_id'] = 223891
afdb_df = afdb_df.loc[afdb_df['taxon_id'] != 369932]

# unique taxon ids in afdb
unique_taxids = afdb_df['taxon_id'].unique()

# load NCBI taxonomy tree
ncbi = NCBITaxa()
tree = ncbi.get_topology(list(unique_taxids))


### Balanced tree paritioning ###

def compute_subtree_sizes(node):
    """ Decorates each node with the size of its subtree. """
    if node.is_leaf():
        node.add_features(subtree_size=1)
    else:
        node.add_features(subtree_size=sum(compute_subtree_sizes(child) for child in node.children) + 1)
    return node.subtree_size

def balanced_clustering(tree, k):
    compute_subtree_sizes(tree) 
    
    clusters = {i: [] for i in range(k)}
    cluster_sizes = [0] * k
    max_cluster_size = (tree.subtree_size + k - 1) // k
    
    def assign_to_cluster(node, cluster_id):
        nonlocal clusters, cluster_sizes
        clusters[cluster_id].append(node.name)
        cluster_sizes[cluster_id] += 1
        
        # if the node has children, try to keep the subtree within the same cluster
        for child in node.children:
            if child.is_leaf() or (cluster_sizes[cluster_id] + child.subtree_size <= max_cluster_size):
                assign_to_cluster(child, cluster_id)
            else:
                next_cluster_id = cluster_sizes.index(min(cluster_sizes))
                assign_to_cluster(child, next_cluster_id)

    assign_to_cluster(tree, cluster_id=0)
    
    return clusters, cluster_sizes

def create_clusters_dict(clusters):
    """ Create a dictionary that maps each identifier to its cluster label. """
    cluster_labels_dict = {}
    for cluster_id, nodes in clusters.items():
        for node in nodes:
            cluster_labels_dict[int(node)] = int(cluster_id)
    return cluster_labels_dict


def k_grouping(k, plot_sizes=False):
    clusters, cluster_sizes = balanced_clustering(tree, k)
    clusters_dict = create_clusters_dict(clusters)
    
    afdb_df.loc[:, f'{k}_grouping'] = afdb_df['taxon_id'].map(clusters_dict)
    
    if plot_sizes:
        plt.boxplot(cluster_sizes)
        plt.title(f'{k}-grouping cluster sizes')
        plt.show()

# Make groupings
        
k_grouping(50, plot_sizes=True)
k_grouping(100, plot_sizes=True)
k_grouping(500, plot_sizes=True)
k_grouping(1000, plot_sizes=True)

afdb_df.to_csv('afdb_with_groupings.csv')