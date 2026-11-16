# usage: Rscript ngbap_run.R X.csv out_prefix [K]; writes <out>_B.csv (direct effects B, B[i,j] != 0: j -> i) and <out>_bi.csv (bidirected adjacency)
args <- commandArgs(TRUE)
.libPaths(c("~/Rlibs", .libPaths())); suppressMessages(library(ngBap))
Y <- as.matrix(read.csv(args[1], header = FALSE)); K <- if (length(args) > 2) as.integer(args[3]) else 4L
out <- bang(Y, K = K, level = 0.01, verbose = FALSE, restrict = 1)
write.table(out$directEffect, paste0(args[2], "_B.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(out$bEdge, paste0(args[2], "_bi.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
