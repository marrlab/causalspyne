import pandas as pd
from causalSpyne.data_linear_gaussian import DataGenLinearGaussian
from causalSpyne.gen_dag_2level import GenDAG2Level
from causalSpyne.dag_gen import GenDAG


def gen_data_linear_gaussian(backbone_num_nodes, backbone_degree,
                             backbone_list_weight_range,
                             num_macro_nodes,
                             num_samples,
                             output_file_names="output.csv"):
    """
    test linear gaussian data gen
    """
    simple_dag_gen = GenDAG(num_nodes=backbone_num_nodes,
                            degree=backbone_degree,
                            list_weight_range=backbone_list_weight_range)
    dag_gen = GenDAG2Level(dag_generator=simple_dag_gen,
                           num_macro_nodes=num_macro_nodes)
    dag = dag_gen.run()
    gen_data = DataGenLinearGaussian(dag)
    arr = gen_data.gen(num_samples=num_samples)
    df = pd.DataFrame(arr,
                      columns=dag.list_node_names)
    df.to_csv(output_file_names, index=False)
    dag.to_binary_csv()


if __name__ == "__main__":
    gen_data_linear_gaussian()

