from scipy.linalg import block_diag
from dag import MatDAG

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
        # iterate over the Macro-DAG edges
        for arc in MatDAG(self.backbone).arcs:
            # macro-DAG node source and sink, (i,j)
            # NP.nonzero
            # iterate all non-zero elements of the DAG adjacency matrix
            macro_arrow_tail, macro_arrow_head = tuple(arc)
            # macro_arrow_tail comes before macro_arrow_head
            node_global_tail = self.dict_cluster_node2dag[str(macro_arrow_tail)].sample_node(macro_arrow_tail)
            node_global_head = self.dict_cluster_node2dag[str(macro_arrow_head)].sample_node(macro_arrow_head)
            # the order is pointing src to sink
            self.fine_grained_dag.add_arc((node_global_tail, node_global_head))

    def run(self):
        self.gen_back_bone()
        self.populate_cluster()
        self.interconnection()
