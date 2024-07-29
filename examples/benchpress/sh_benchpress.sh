cp output.csv benchpress/resources/data/mydatasets/

cp examples/benchpress/causalSpyne.json benchpress/config/causalSpyne.json

cp examples/benchpress/causalSpyne_dev.json benchpress/config/causalSpyne.json


cp examples/benchpress/full_dev.json benchpress/config/causalSpyne.json


cp adj.csv benchpress/resources/adjmat/myadjmats/adj.csv

touch benchpress/resources/constraints/None-pcalg
touch benchpress/resources/constraints/None-bnlearn

cd benchpress
git checkout dev

snakemake --cores all --use-singularity --configfile config/causalSpyne.json


cd results/output/causalSpyne/benchmarks
