# install.packages("devtools")
# devtools::install_github("smilesun/ngBap")
dat <- ngBap::rBAP(n = 50000, p = 7, dist = "gamma", d = 3, b = 5, ancestral = F, shuffle = T, signs = T)
Y <- dat$Y
out <- ngBap::bang(Y, K = 3, level = .01, verbose = F, restrict = 1)
