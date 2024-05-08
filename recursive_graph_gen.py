

class RecursiveGraphGen():
    def __init__(self, dag_generator, backbone_density,
                 strategy_num_nodes_per_cluster, n_cluster):
        self.dag_generator = dag_generator
        self.backbone = None
        self.num_clusters = n_cluster
        self.backbone_density = backbone_density
        self.strategy_num_nodes_per_cluster = strategy_num_nodes_per_cluster
        self.num_nodes_per_cluster = self.strategy_num_nodes_per_cluster(
            self.num_clusters)
        self.dict_cluster_node2dag = {}
        self.dag = None

    def gen_back_bone(self):
        self.backbone = self.dag_generator.genDAG(
            self.num_clusters, self.backbone_density)

    def populate_cluster(self):
        for node in self.backbone:
            self.dict_cluster_node2dag[node] = self.dag_generator.genDAG(
                self.num_nodes_per_cluster[node])

    def interconnection(self):
        self.backbone.toplogical_order()
        for arc in self.backbone.arcs():
            cluster1, cluster2 = arc.nodes_pair()
            # cluster1 comes before cluster2
            node1 = self.dict_cluster_node2dag[cluster1].sample_node()
            node2 = self.dict_cluster_node2dag[cluster2].sample_node()
            self.dag.add_arc((node1, node2))
