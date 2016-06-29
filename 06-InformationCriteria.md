06 - Information Criteria
================
TJ Mahr
June 29, 2016

Preamble
--------

This is notebook of code I wrote while reading Chapter 6 of [*Statistical Rethinking*](http://xcelab.net/rm/statistical-rethinking/). I don't use the author's [rethinking package](https://github.com/rmcelreath/rethinking) or any of its helper functions. I instead used rstanarm to fit the Bayesian models and loo for model comparisons, because I'm trying to stay in that ecosystem. As a result, my models have different priors and different parameter estimates.

``` r
# devtools::install_github("rmcelreath/rethinking")
library("dplyr", warn.conflicts = FALSE)
#> Warning: package 'dplyr' was built under R version 3.3.1
library("rstanarm")
#> Warning: package 'rstanarm' was built under R version 3.3.1
#> Loading required package: Rcpp
#> rstanarm (Version 2.10.1, packaged: 2016-06-24 17:54:22 UTC)
#> - Do not expect the default priors to remain the same in future rstanarm versions.
#> Thus, R scripts should specify priors explicitly, even if they are just the defaults.
#> - For execution on a local, multicore CPU with excess RAM we recommend calling
#> options(mc.cores = parallel::detectCores())
library("purrr")
#> 
#> Attaching package: 'purrr'
#> The following objects are masked from 'package:dplyr':
#> 
#>     contains, order_by
library("tidyr")
library("ggplot2")

# I don't load the rethinking package, but I use data from it. Log version.
rethinking_info <- packageDescription(
  pkg = "rethinking", 
  fields = c("Version", "GithubUsername", "GithubRepo", 
             "GithubRef", "GithubSHA1"))
str(rethinking_info, give.attr = FALSE)
#> List of 5
#>  $ Version       : chr "1.59"
#>  $ GithubUsername: chr "rmcelreath"
#>  $ GithubRepo    : chr "rethinking"
#>  $ GithubRef     : chr "master"
#>  $ GithubSHA1    : chr "a309712d904d1db7af1e08a76c521ab994006fd5"
```

Fit the models
--------------

Get the primate milk data. The example models will predict the kilocalories per gram of milk using the mass of the mother (log kg) and the proportion of the brain that is neocortex. These data are from Hinde and Milligan (2011), according to `?milk`.

``` r
data(milk, package = "rethinking")
d <- milk %>% 
  as_data_frame %>% 
  filter(!is.na(neocortex.perc)) %>% 
  mutate(neocortex = neocortex.perc / 100,
         log_mass = log(mass)) %>% 
  select(clade, species, kcal.per.g, mass, log_mass, neocortex)
d %>% knitr::kable(digits = 3)
```

| clade            | species                 |  kcal.per.g|   mass|  log\_mass|  neocortex|
|:-----------------|:------------------------|-----------:|------:|----------:|----------:|
| Strepsirrhine    | Eulemur fulvus          |        0.49|   1.95|      0.668|      0.552|
| New World Monkey | Alouatta seniculus      |        0.47|   5.25|      1.658|      0.645|
| New World Monkey | A palliata              |        0.56|   5.37|      1.681|      0.645|
| New World Monkey | Cebus apella            |        0.89|   2.51|      0.920|      0.676|
| New World Monkey | S sciureus              |        0.92|   0.68|     -0.386|      0.688|
| New World Monkey | Cebuella pygmaea        |        0.80|   0.12|     -2.120|      0.588|
| New World Monkey | Callimico goeldii       |        0.46|   0.47|     -0.755|      0.617|
| New World Monkey | Callithrix jacchus      |        0.71|   0.32|     -1.139|      0.603|
| Old World Monkey | Miopithecus talpoin     |        0.68|   1.55|      0.438|      0.700|
| Old World Monkey | M mulatta               |        0.97|   3.24|      1.176|      0.704|
| Old World Monkey | Papio spp               |        0.84|  12.30|      2.510|      0.734|
| Ape              | Hylobates lar           |        0.62|   5.37|      1.681|      0.675|
| Ape              | Pongo pygmaeus          |        0.54|  35.48|      3.569|      0.713|
| Ape              | Gorilla gorilla gorilla |        0.49|  79.43|      4.375|      0.726|
| Ape              | Pan paniscus            |        0.48|  40.74|      3.707|      0.702|
| Ape              | P troglodytes           |        0.55|  33.11|      3.500|      0.763|
| Ape              | Homo sapiens            |        0.71|  54.95|      4.006|      0.755|

Fit four different models that we will compare later.

``` r
# Intercept only
m1 <- stan_glm(
  formula = kcal.per.g ~ 1,
  data = d,
  family = gaussian(),
  prior = normal(0, 1),
  prior_intercept = normal(.5, .5),
  prior_ops = prior_options(prior_scale_for_dispersion = .25)
)
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 1).
#> 
#> Chain 1, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 1, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 1, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 1, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 1, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 1, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 1, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 1, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 1, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 1, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 1, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 1, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.02 seconds (Warm-up)
#>                0.024 seconds (Sampling)
#>                0.044 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 2).
#> 
#> Chain 2, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 2, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 2, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 2, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 2, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 2, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 2, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 2, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 2, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 2, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 2, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 2, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.02 seconds (Warm-up)
#>                0.027 seconds (Sampling)
#>                0.047 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 3).
#> 
#> Chain 3, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 3, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 3, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 3, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 3, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 3, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 3, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 3, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 3, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 3, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 3, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 3, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.019 seconds (Warm-up)
#>                0.024 seconds (Sampling)
#>                0.043 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 4).
#> 
#> Chain 4, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 4, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 4, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 4, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 4, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 4, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 4, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 4, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 4, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 4, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 4, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 4, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.02 seconds (Warm-up)
#>                0.025 seconds (Sampling)
#>                0.045 seconds (Total)

# One predictor
m2 <- update(m1, . ~ neocortex)
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 1).
#> 
#> Chain 1, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 1, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 1, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 1, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 1, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 1, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 1, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 1, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 1, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 1, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 1, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 1, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.044 seconds (Warm-up)
#>                0.043 seconds (Sampling)
#>                0.087 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 2).
#> 
#> Chain 2, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 2, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 2, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 2, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 2, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 2, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 2, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 2, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 2, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 2, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 2, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 2, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.043 seconds (Warm-up)
#>                0.042 seconds (Sampling)
#>                0.085 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 3).
#> 
#> Chain 3, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 3, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 3, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 3, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 3, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 3, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 3, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 3, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 3, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 3, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 3, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 3, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.046 seconds (Warm-up)
#>                0.046 seconds (Sampling)
#>                0.092 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 4).
#> 
#> Chain 4, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 4, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 4, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 4, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 4, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 4, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 4, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 4, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 4, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 4, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 4, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 4, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.042 seconds (Warm-up)
#>                0.043 seconds (Sampling)
#>                0.085 seconds (Total)
m3 <- update(m1, . ~ log_mass)
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 1).
#> 
#> Chain 1, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 1, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 1, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 1, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 1, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 1, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 1, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 1, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 1, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 1, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 1, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 1, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.04 seconds (Warm-up)
#>                0.042 seconds (Sampling)
#>                0.082 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 2).
#> 
#> Chain 2, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 2, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 2, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 2, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 2, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 2, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 2, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 2, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 2, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 2, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 2, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 2, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.043 seconds (Warm-up)
#>                0.04 seconds (Sampling)
#>                0.083 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 3).
#> 
#> Chain 3, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 3, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 3, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 3, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 3, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 3, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 3, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 3, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 3, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 3, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 3, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 3, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.042 seconds (Warm-up)
#>                0.044 seconds (Sampling)
#>                0.086 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 4).
#> 
#> Chain 4, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 4, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 4, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 4, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 4, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 4, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 4, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 4, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 4, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 4, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 4, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 4, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.043 seconds (Warm-up)
#>                0.041 seconds (Sampling)
#>                0.084 seconds (Total)

# Two predictors
m4 <- update(m1, . ~ neocortex + log_mass)
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 1).
#> 
#> Chain 1, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 1, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 1, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 1, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 1, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 1, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 1, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 1, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 1, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 1, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 1, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 1, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.07 seconds (Warm-up)
#>                0.067 seconds (Sampling)
#>                0.137 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 2).
#> 
#> Chain 2, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 2, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 2, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 2, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 2, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 2, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 2, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 2, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 2, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 2, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 2, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 2, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.064 seconds (Warm-up)
#>                0.06 seconds (Sampling)
#>                0.124 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 3).
#> 
#> Chain 3, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 3, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 3, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 3, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 3, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 3, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 3, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 3, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 3, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 3, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 3, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 3, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.065 seconds (Warm-up)
#>                0.062 seconds (Sampling)
#>                0.127 seconds (Total)
#> 
#> 
#> SAMPLING FOR MODEL 'continuous' NOW (CHAIN 4).
#> 
#> Chain 4, Iteration:    1 / 2000 [  0%]  (Warmup)
#> Chain 4, Iteration:  200 / 2000 [ 10%]  (Warmup)
#> Chain 4, Iteration:  400 / 2000 [ 20%]  (Warmup)
#> Chain 4, Iteration:  600 / 2000 [ 30%]  (Warmup)
#> Chain 4, Iteration:  800 / 2000 [ 40%]  (Warmup)
#> Chain 4, Iteration: 1000 / 2000 [ 50%]  (Warmup)
#> Chain 4, Iteration: 1001 / 2000 [ 50%]  (Sampling)
#> Chain 4, Iteration: 1200 / 2000 [ 60%]  (Sampling)
#> Chain 4, Iteration: 1400 / 2000 [ 70%]  (Sampling)
#> Chain 4, Iteration: 1600 / 2000 [ 80%]  (Sampling)
#> Chain 4, Iteration: 1800 / 2000 [ 90%]  (Sampling)
#> Chain 4, Iteration: 2000 / 2000 [100%]  (Sampling)
#>  Elapsed Time: 0.068 seconds (Warm-up)
#>                0.064 seconds (Sampling)
#>                0.132 seconds (Total)
```

Compare posterior to prior in each model.

``` r
# We want to have the same color-parameter mapping across plots, otherwise the
# sigma term will have three different colors. Manually set colors, using the D3
# color palette
d3_colors <- c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")
p1 <- posterior_vs_prior(m1) + scale_color_manual(values = d3_colors[c(1, 4)])
#> 
#> Drawing from prior...
#> 
#>  Elapsed Time: 0.02 seconds (Warm-up)
#>                0.026 seconds (Sampling)
#>                0.046 seconds (Total)
#> 
#> 
#>  Elapsed Time: 0.02 seconds (Warm-up)
#>                0.024 seconds (Sampling)
#>                0.044 seconds (Total)
p2 <- posterior_vs_prior(m2) + scale_color_manual(values = d3_colors[c(1, 3:4)])
#> 
#> Drawing from prior...
#> 
#>  Elapsed Time: 0.043 seconds (Warm-up)
#>                0.045 seconds (Sampling)
#>                0.088 seconds (Total)
#> 
#> 
#>  Elapsed Time: 0.043 seconds (Warm-up)
#>                0.042 seconds (Sampling)
#>                0.085 seconds (Total)
p3 <- posterior_vs_prior(m3) + scale_color_manual(values = d3_colors[c(1:2, 4)])
#> 
#> Drawing from prior...
#> 
#>  Elapsed Time: 0.056 seconds (Warm-up)
#>                0.065 seconds (Sampling)
#>                0.121 seconds (Total)
#> 
#> 
#>  Elapsed Time: 0.051 seconds (Warm-up)
#>                0.055 seconds (Sampling)
#>                0.106 seconds (Total)
p4 <- posterior_vs_prior(m4) + scale_color_manual(values = d3_colors[c(1:4)])
#> 
#> Drawing from prior...
#> 
#>  Elapsed Time: 0.053 seconds (Warm-up)
#>                0.048 seconds (Sampling)
#>                0.101 seconds (Total)
#> 
#> 
#>  Elapsed Time: 0.049 seconds (Warm-up)
#>                0.051 seconds (Sampling)
#>                0.1 seconds (Total)
cowplot::plot_grid(p1, p2, p3, p4)
```

![](06-InformationCriteria_files/figure-markdown_github/posterior-vs-prior-1.png)

Manual AIC calculation
----------------------

To figure out what these information criteria are measuring, I want to calculate them by hand.

The AIC works on the classical models, so I write a helper function to convert an RStanArm glm model into a classical glm model.

``` r
# Refit a stanreg model using glm. Assumes no other glm arguments are used 
# besides formula, data and family--i.e., this is doesn't work for all glm
# models.
as_glm <- function(stan_model) {
  glm(stan_model$formula, data = stan_model$data, family = stan_model$family)
}

glm_m4 <- as_glm(m4)
```

Plus, another helper to get the maximum likelihood estimate of the model's sigma term.

``` r
# Get the maximum likelihood estimate of sigma from a model
# http://stats.stackexchange.com/a/73245
# https://stat.ethz.ch/pipermail/r-help/2003-June/035518.html
sigma_ml <- function(m) {
  squared_errors <- residuals(m) ^ 2
  sqrt(mean(squared_errors))
}
```

According to our model, each observation falls on a normal curve with a mean equal to fitted value, and a standard deviation equal to model sigma. The density of the fitted-value curve at the observed value is the likelihood of the data.

To compute the likelihood of each observation, we find the density of observations on their fitted-value curves. These are three such densities. (Code omitted because it's hairy base-graphics code built from the example code in `?dnorm`.)

![](06-InformationCriteria_files/figure-markdown_github/likelihood%20curves-1.png)

The sum of the log of the likelihoods is therefore the log-likelihood of a model.

``` r
# By hand calculation
log_likelihoods <- dnorm(
  x = glm_m4$y, 
  mean = glm_m4$fitted.values, 
  sd = sigma_ml(glm_m4), 
  log = TRUE)
log_likelihoods
#>           1           2           3           4           5           6 
#>  0.87397004  0.95498445  1.24492865  0.08700352  1.16873143  1.19341071 
#>           7           8           9          10          11          12 
#> -1.13692008  1.24573233  0.42789805 -0.30246193  0.72560506  1.23223891 
#>          13          14          15          16          17 
#>  1.22896834  1.20983370  1.18740026  0.29284919  1.04353561

sum(log_likelihoods)
#> [1] 12.67771

# Automatic calculation
logLik(glm_m4)
#> 'log Lik.' 12.67771 (df=4)

# Because residuals have mean 0, we can just use those instead of fitted values
dnorm(resid(glm_m4), mean = 0, sd = sigma_ml(glm_m4), log = TRUE) %>% sum
#> [1] 12.67771
```

From the log-likelihood, we can compute deviance, AIC, and BIC.

``` r
# as.numeric to get rid of df attribute
deviance_m4 <- as.numeric(-2 * logLik(glm_m4))
npars_m4 <- attr(logLik(glm_m4), "df")

deviance_m4
#> [1] -25.35542
```

The information criteria are the deviance measures plus some penalty *k* times the number of parameters. For AIC, the penalty is *k*=2. For BIC, the penalty is the log of the number of observations, *k*=log(*n*).

``` r
# Manual AIC vs automatic
aic_m4 <- deviance_m4 + 2 * npars_m4
aic_m4
#> [1] -17.35542

AIC(glm_m4)  
#> [1] -17.35542

# Manual BIC vs automatic  
bic_m4 <- deviance_m4 + log(nobs(glm_m4)) * npars_m4
bic_m4
#> [1] -14.02256

BIC(glm_m4)
#> [1] -14.02256
```

Manual WAIC calculation
-----------------------

The above AIC example measured one log-likelihood based on one set of parameters. Our Bayesian models reflect a distribution of parameter estimates, and we sampled thousands of parameter estimates from the posterior distribution. As a result, we have a distribution of log-likelihoods, and the posterior information criteria will incorporate information from the distribution of log-likelihoods.

The WAIC (Widely Applicable Information Criterion) is calculated using pointwise log-likelihoods. Specifically, we calculate the log-pointwise-predictive density (lppd). We observation has its own likelihood in each model, so we calculate each observation's average likelihood across models and take the log.

RStanARM provides a `log_lik` to get an n\_samples x n\_observations matrix of log-likelihoods.

``` r
dim(log_lik(m4))
#> [1] 4000   17

# log-pointwise-predictive density as the log of the average
# likelihood of each oberservation
each_lppd <- log_lik(m4) %>% exp %>% colMeans %>% log 
each_lppd
#>  [1]  0.8062159  0.7967450  1.0514920  0.1039733  0.8627891  0.9160301
#>  [7] -0.4056730  1.0133188  0.6327011 -0.2399497  0.5741003  1.0618466
#> [13]  1.0204555  0.9692484  0.9638801  0.4813494  0.8560871
```

We also need a penalty term. Here, it's the effective number of parameters (p\_waic). Each observation contributes to the penalty term, using the variance of its log-likelihoods.

``` r
# effective number of parameters by taking variance of log-likelihood of each
# observation 
each_p_waic <- log_lik(m4) %>% apply(2, var)
each_p_waic
#>  [1] 0.26049375 0.06440188 0.04368429 0.15748117 0.12226692 0.11350211
#>  [7] 0.78082479 0.05645529 0.15015606 0.35433094 0.12800578 0.03882088
#> [13] 0.04720713 0.06578667 0.05992431 0.26185287 0.09746762
```

The WAIC is the difference in total lppd and total p\_waic on the deviance scale.

``` r
lppd <- sum(each_lppd)
lppd
#> [1] 11.46461
p_waic <- sum(each_p_waic)
p_waic
#> [1] 2.802662
waic <- -2 * (lppd - p_waic)
waic
#> [1] -17.32389

# skip the summing step
each_waic <- -2 * (each_lppd - each_p_waic)
sum(each_waic)
#> [1] -17.32389
```

But because each point contributes information to the lppd and p\_waic calculations, we can compute a standard error on these numbers.

``` r
# standard error of the waics
se <- function(xs) sqrt(length(xs) * var(xs))
se(each_p_waic)
#> [1] 0.751527
se(each_waic)
#> [1] 5.045902
```

These calculations match the estimates from the loo package. The loo::waic estimates use expected lppd (elpd) which already subtracts p\_waic from lppd. It's *expected* because it is the log-likelihood of the in-sample data adjusted with a penalty term. The resulting value is an expectation for out-of-sample data.

``` r
# manual elpd_waic
lppd - p_waic
#> [1] 8.661947
se(each_lppd - each_p_waic)
#> [1] 2.522951

loo::waic(m4)
#> Computed from 4000 by 17 log-likelihood matrix
#> 
#>           Estimate  SE
#> elpd_waic      8.7 2.5
#> p_waic         2.8 0.8
#> waic         -17.3 5.0
#> Warning: 1 (5.9%) p_waic estimates greater than 0.4.
#> We recommend trying loo() instead.
```

Getting WAIC for the models
---------------------------

Get the WAIC using the loo package.

I'm going to experiment with the fitting-many-models, data-frame-of-models technique here. That means doing some unconventional things with data-frames.

Basically, I make a data-frame with one row per model and then do things to each model, storing the results of those operations as new columns in the data-frame. Specifically, I calculate the WAIC of the Stan models and refit the models using the classical technique so I can get an AIC value to compare to the WAIC value.

``` r
# Get a data-frame summary from a loo::waic object
tidy_waic <- function(waic_fit) {
  to_keep <- c("waic", "se_waic", "elpd_waic", "se_elpd_waic", 
               "p_waic", "se_p_waic")
  ret <- waic_fit[to_keep]
  as_data_frame(ret)
}

# Create a data-frame with one-row per model
model_summary <- data_frame(StanFit = list(m1, m2, m3, m4)) %>% 
  mutate(
    # Use the formula from the model as a label
    Formula = map_chr(StanFit, ~ .x$formula %>% deparse),
    # Get the classical fit and deviance/AIC of classical fit
    ClassicalFit = map(StanFit, as_glm),
    Deviance = map_dbl(ClassicalFit, ~ logLik(.x) * -2),
    AIC = map_dbl(ClassicalFit, AIC),
    # Get the WAIC summary from the Stan fit of the model
    WAIC = map(StanFit, . %>% loo::waic() %>% tidy_waic)) %>% 
  # The WAIC is stored as a data-frame within each row. Unnest to promote the 
  # columns from the nested data-frame
  unnest(WAIC)

# We need to exclude the list columns in order to print a formatted table 
model_summary %>% 
  select(-StanFit, -ClassicalFit) %>% 
  arrange(waic) %>% 
  knitr::kable(, digits = 2) 
```

| Formula                            |  Deviance|     AIC|    waic|  se\_waic|  elpd\_waic|  se\_elpd\_waic|  p\_waic|  se\_p\_waic|
|:-----------------------------------|---------:|-------:|-------:|---------:|-----------:|---------------:|--------:|------------:|
| kcal.per.g ~ neocortex + log\_mass |    -25.36|  -17.36|  -17.32|      5.05|        8.66|            2.52|     2.80|         0.75|
| kcal.per.g ~ log\_mass             |    -14.74|   -8.74|   -9.59|      4.44|        4.80|            2.22|     1.84|         0.41|
| kcal.per.g ~ 1                     |    -12.46|   -8.46|   -9.01|      4.10|        4.51|            2.05|     1.25|         0.31|
| kcal.per.g ~ neocortex             |    -12.87|   -6.87|   -7.84|      3.68|        3.92|            1.84|     1.80|         0.31|
