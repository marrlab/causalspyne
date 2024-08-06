cp output.csv benchpress/resources/data/mydatasets/

cp examples/benchpress/causalspyne.json benchpress/config/causalspyne.json

cp examples/benchpress/causalspyne_dev.json benchpress/config/causalspyne.json


cp examples/benchpress/full_dev.json benchpress/config/causalspyne.json


cp adj.csv benchpress/resources/adjmat/myadjmats/adj.csv

# touch benchpress/resources/constraints/None-pcalg
echo "{}" > benchpress/resources/constraints/None-pcalg
# touch benchpress/resources/constraints/None-bnlearn

cd benchpress
git checkout dev
git checkout b2c482fa7089980eaa8769b55b2c2d1dad9c2dc1

snakemake --cores all --use-singularity --configfile config/causalspyne.json


cd results/output/causalspyne/benchmarks
