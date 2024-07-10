from recursive_graph_gen import RecursiveGraphGen, NumNodesPerCluster
from dag_gen import Erdos_Renyi, GenDAGER



def test_recursive_gen():

    dag_gen = GenDAGER(num_nodes=3, degree=2, weight_range=[3, 5])
    getter_num_nodes = NumNodesPerCluster()
    gen = RecursiveGraphGen(dag_generator=dag_gen,
                      backbone_density=0.7,
                      strategy_num_nodes_per_cluster=getter_num_nodes,
                      n_cluster=3)
    gen.run()
