import pandas as pd
from causalSpyne.data_gen import DataGen
from causalSpyne.gen_dag_2level import GenDAG2Level
from causalSpyne.dag_gen import GenDAG


def gen_partially_observed_data(backbone_num_nodes, backbone_degree,
                                backbone_list_weight_range,
                                num_macro_nodes,
                                num_samples,
                                output_file_names="output.csv"):
    """
    test linear gaussian data gen
    """


    simple_dag_gen = GenDAG(num_nodes=4, degree=2, list_weight_range=[3, 5])

    dag_gen = GenDAG2Level(dag_generator=simple_dag_gen, num_macro_nodes=4)

    dag = dag_gen.run()
    dag.visualize(title="dag_complete_confounder")
    dag.to_binary_csv()
    gen_data = DataGen(dag)
    gen_data.gen(num_samples=200)
    gen_data.to_csv()

    subview = DAGView(dag=dag)
    subview.run(num_samples=200, confound=True, list_nodes2hide=[0.99])
    subview.to_csv()
    subview.visualize(title="dag_marginal_confounder")
