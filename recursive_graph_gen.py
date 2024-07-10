from scipy.linalg import block_diag

class ClassBackBone():
    """
    """
    @property
    def arcs(self):
        """
        """

class NumNodesPerCluster():
    def __call__(self, num_clusters):
        return 3


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
        self.fine_grained_dag = None

    def gen_back_bone(self):
        self.backbone = self.dag_generator.genDAG(
             self.num_clusters)
        # FIXME: self.num_clusters, self.backbone_density)

    def get_dag_size(self, dag):
        """
        """
        return dag.shape[0]

    def populate_cluster(self):
        """
        replace a macro node into a DAG
        """
        num_nodes = 0
        # iterate each macro node
        # for (i, _) in enumerate(self.backbone):
        for i in range(self.backbone.shape[0]):
            self.dict_cluster_node2dag[str(i)] = self.dag_generator.genDAG(3)
            # num_nodes += self.get_dag_size(self.dict_cluster_node2dag[node])
        self.init_fine_grained()   # block diagnoal

    def init_fine_grained(self):
        self.fine_grained_dag = block_diag(
            *(tuple(self.dict_cluster_node2dag.values())))

    def interconnection(self):
        self.backbone.toplogical_order()
        for arc in self.backbone.arcs():
            str_cluster_src, str_clustr_sink = arc.nodes_pair()
            # str_cluster_src comes before str_clustr_sink
            node_micro_src = self.dict_cluster_node2dag[str_cluster_src].sample_node()
            node_micro_sink = self.dict_cluster_node2dag[str_clustr_sink].sample_node()
            # the order is pointing src to sink
            self.fine_grained_dag.add_arc((node_micro_src, node_micro_sink))

    def run(self):
        self.gen_back_bone()
        self.populate_cluster()
        self.interconnection()
