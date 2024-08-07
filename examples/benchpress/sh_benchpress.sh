
cp data_subdag.csv submodules/benchpress/resources/data/mydatasets/
cp adj.csv submodules/benchpress/resources/adjmat/myadjmats/adj.csv

# cp submodules/benchpress/config/paper_sachs.json examples/benchpress/causalspyne_master.json

cp examples/benchpress/causalspyne_master.json submodules/benchpress/config/causalspyne.json


#cp examples/benchpress/causalspyne.json submodules/benchpress/config/causalspyne.json

#cp examples/benchpress/causalspyne_dev.json submodules/benchpress/config/causalspyne.json

#cp examples/benchpress/full_dev.json submodules/benchpress/config/causalspyne.json



# touch benchpress/resources/constraints/None-pcalg
# echo "{}" > submodules/benchpress/resources/constraints/None-pcalg
# touch benchpress/resources/constraints/None-bnlearn

cd submodules/benchpress
# git checkout dev
# git checkout b2c482fa7089980eaa8769b55b2c2d1dad9c2dc1

snakemake --cores all --use-singularity --configfile config/causalspyne.json


cd results/output/causalspyne/benchmarks
