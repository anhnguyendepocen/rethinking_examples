06 - Information Criteria
================
TJ Mahr
June 29, 2016

Preamble
--------

This is notebook of code I wrote while reading Chapter 6 of [*Statistical Rethinking*](http://xcelab.net/rm/statistical-rethinking/). I don't use the author's [rethinking package](https://github.com/rmcelreath/rethinking) or any of its helper functions. I instead used [RStanARM](http://mc-stan.org/%20interfaces/rstanarm) to fit the Bayesian models and [Loo](http://mc-stan.org/%20interfaces/loo.html) for model comparisons, because I'm trying to stay in that ecosystem. As a result, my models have different priors and different parameter estimates.

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
library("purrr", warn.conflicts = FALSE)
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

Prepare the primate milk data
-----------------------------

The example models will predict the kilocalories per gram of milk using the mass of the mother (log kg) and the proportion of the brain that is neocortex. These data are from Hinde and Milligan (2011), according to `?milk`.

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

Fit the models
--------------

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

# One predictor
m2 <- update(m1, . ~ neocortex)
m3 <- update(m1, . ~ log_mass)

# Two predictors
m4 <- update(m1, . ~ neocortex + log_mass)
```

``` r
summary(m1)
#> stan_glm(formula = kcal.per.g ~ 1, family = gaussian(), data = d, 
#>     prior = normal(0, 1), prior_intercept = normal(0.5, 0.5), 
#>     prior_ops = prior_options(prior_scale_for_dispersion = 0.25))
#> 
#> Family: gaussian (identity)
#> Algorithm: sampling
#> Posterior sample size: 4000
#> Observations: 17
#> 
#> Estimates:
#>                 mean   sd   2.5%   25%   50%   75%   97.5%
#> (Intercept)   0.6    0.0  0.6    0.6   0.6   0.7   0.7    
#> sigma         0.2    0.0  0.1    0.2   0.2   0.2   0.3    
#> mean_PPD      0.6    0.1  0.5    0.6   0.6   0.7   0.8    
#> log-posterior 3.8    1.0  1.0    3.4   4.1   4.5   4.7    
#> 
#> Diagnostics:
#>               mcse Rhat n_eff
#> (Intercept)   0.0  1.0  2892 
#> sigma         0.0  1.0  2046 
#> mean_PPD      0.0  1.0  3214 
#> log-posterior 0.0  1.0  1447 
#> 
#> For each parameter, mcse is Monte Carlo standard error, n_eff is a crude measure of effective sample size, and Rhat is the potential scale reduction factor on split chains (at convergence Rhat=1).
summary(m2)
#> stan_glm(formula = kcal.per.g ~ neocortex, family = gaussian(), 
#>     data = d, prior = normal(0, 1), prior_intercept = normal(0.5, 
#>         0.5), prior_ops = prior_options(prior_scale_for_dispersion = 0.25))
#> 
#> Family: gaussian (identity)
#> Algorithm: sampling
#> Posterior sample size: 4000
#> Observations: 17
#> 
#> Estimates:
#>                 mean   sd   2.5%   25%   50%   75%   97.5%
#> (Intercept)    0.4    0.5 -0.6    0.0   0.4   0.7   1.3   
#> neocortex      0.4    0.7 -1.0   -0.1   0.4   0.9   1.9   
#> sigma          0.2    0.0  0.1    0.2   0.2   0.2   0.3   
#> mean_PPD       0.6    0.1  0.5    0.6   0.6   0.7   0.8   
#> log-posterior  2.4    1.3 -0.8    1.8   2.8   3.3   3.9   
#> 
#> Diagnostics:
#>               mcse Rhat n_eff
#> (Intercept)   0.0  1.0  2845 
#> neocortex     0.0  1.0  2849 
#> sigma         0.0  1.0  2704 
#> mean_PPD      0.0  1.0  3359 
#> log-posterior 0.0  1.0  1859 
#> 
#> For each parameter, mcse is Monte Carlo standard error, n_eff is a crude measure of effective sample size, and Rhat is the potential scale reduction factor on split chains (at convergence Rhat=1).
summary(m3)
#> stan_glm(formula = kcal.per.g ~ log_mass, family = gaussian(), 
#>     data = d, prior = normal(0, 1), prior_intercept = normal(0.5, 
#>         0.5), prior_ops = prior_options(prior_scale_for_dispersion = 0.25))
#> 
#> Family: gaussian (identity)
#> Algorithm: sampling
#> Posterior sample size: 4000
#> Observations: 17
#> 
#> Estimates:
#>                 mean   sd   2.5%   25%   50%   75%   97.5%
#> (Intercept)    0.7    0.1  0.6    0.7   0.7   0.7   0.8   
#> log_mass       0.0    0.0 -0.1    0.0   0.0   0.0   0.0   
#> sigma          0.2    0.0  0.1    0.2   0.2   0.2   0.3   
#> mean_PPD       0.7    0.1  0.5    0.6   0.7   0.7   0.8   
#> log-posterior  3.3    1.3  0.0    2.7   3.6   4.2   4.8   
#> 
#> Diagnostics:
#>               mcse Rhat n_eff
#> (Intercept)   0.0  1.0  2771 
#> log_mass      0.0  1.0  3016 
#> sigma         0.0  1.0  2263 
#> mean_PPD      0.0  1.0  3259 
#> log-posterior 0.0  1.0  1729 
#> 
#> For each parameter, mcse is Monte Carlo standard error, n_eff is a crude measure of effective sample size, and Rhat is the potential scale reduction factor on split chains (at convergence Rhat=1).
summary(m4)
#> stan_glm(formula = kcal.per.g ~ neocortex + log_mass, family = gaussian(), 
#>     data = d, prior = normal(0, 1), prior_intercept = normal(0.5, 
#>         0.5), prior_ops = prior_options(prior_scale_for_dispersion = 0.25))
#> 
#> Family: gaussian (identity)
#> Algorithm: sampling
#> Posterior sample size: 4000
#> Observations: 17
#> 
#> Estimates:
#>                 mean   sd   2.5%   25%   50%   75%   97.5%
#> (Intercept)   -0.8    0.5 -1.8   -1.2  -0.8  -0.5   0.2   
#> neocortex      2.4    0.8  0.7    1.9   2.4   2.9   3.9   
#> log_mass      -0.1    0.0 -0.1   -0.1  -0.1  -0.1   0.0   
#> sigma          0.1    0.0  0.1    0.1   0.1   0.2   0.2   
#> mean_PPD       0.7    0.0  0.6    0.6   0.7   0.7   0.7   
#> log-posterior  6.1    1.5  2.2    5.4   6.4   7.2   8.0   
#> 
#> Diagnostics:
#>               mcse Rhat n_eff
#> (Intercept)   0.0  1.0  1833 
#> neocortex     0.0  1.0  1776 
#> log_mass      0.0  1.0  1885 
#> sigma         0.0  1.0  1835 
#> mean_PPD      0.0  1.0  2680 
#> log-posterior 0.0  1.0  1478 
#> 
#> For each parameter, mcse is Monte Carlo standard error, n_eff is a crude measure of effective sample size, and Rhat is the potential scale reduction factor on split chains (at convergence Rhat=1).
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
p2 <- posterior_vs_prior(m2) + scale_color_manual(values = d3_colors[c(1, 3:4)])
#> 
#> Drawing from prior...
p3 <- posterior_vs_prior(m3) + scale_color_manual(values = d3_colors[c(1:2, 4)])
#> 
#> Drawing from prior...
p4 <- posterior_vs_prior(m4) + scale_color_manual(values = d3_colors[c(1:4)])
#> 
#> Drawing from prior...
```

``` r
cowplot::plot_grid(p1, p2, p3, p4)
```

![](06-InformationCriteria_files/figure-markdown_github/posterior-vs-prior-1.png)

Model comparison with WAIC says the fourth one is to be preferred. Why?

``` r
loo::compare(loo::waic(m1), loo::waic(m2), loo::waic(m3), loo::waic(m4))
#>               waic  se_waic elpd_waic se_elpd_waic p_waic se_p_waic
#> loo::waic(m4) -17.7   5.0     8.8       2.5          2.7    0.7    
#> loo::waic(m3)  -9.5   4.5     4.7       2.2          1.9    0.4    
#> loo::waic(m1)  -9.1   4.1     4.6       2.0          1.2    0.3    
#> loo::waic(m2)  -7.7   3.6     3.8       1.8          1.8    0.3
```

Manual AIC calculation
----------------------

Deviance measures relative model fit using the log-likelihood of the data. The information criteria adjust the deviance to make a prediction about the model's deviance given some new out-of-sample data. These information criterion therefore estimate a model's relative predictive accurary.

To figure out what the information criteria are measuring, I want to calculate them by hand. The AIC works on classical models, so I wrote a helper function to convert an RStanArm glm model into a classical glm model.

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

To compute the likelihood of each observation, we find the density of observations on their fitted-value curves. These are three such densities from model 4. (Code omitted because it's hairy base-graphics code built from the example code in `?dnorm`.)

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

The information criteria are the deviance measures plus some penalty *k* times the number of parameters. For AIC, the penalty is *k*=2. For BIC, the penalty is the log of the number of observations, *k*=log(*n*). The penalty adjusts the in- sample deviance to get predicted value of the out-of-sample deviance. The model with the lowest predicted out-of-sample deviance is the favored model.

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

The above AIC example measured one log-likelihood based on one set of parameters. Our Bayesian model reflects a distribution of parameter estimates, and we sampled thousands of parameter estimates from the posterior distribution. As a result, we have a distribution of log-likelihoods, and posterior information criteria will incorporate information from the distribution of log-likelihoods.

The WAIC (Widely Applicable Information Criterion) is calculated using pointwise log-likelihoods. Specifically, we calculate the log-pointwise-predictive density (lppd). Each observation has its own likelihood in each model, so we calculate each observation's average likelihood across models and take the log.

RStanARM provides a `log_lik` to get an n\_samples x n\_observations matrix of log-likelihoods.

``` r
dim(log_lik(m4))
#> [1] 4000   17

# log-pointwise-predictive density as the log of the average
# likelihood of each oberservation
each_lppd <- log_lik(m4) %>% exp %>% colMeans %>% log 
each_lppd
#>  [1]  0.82571372  0.79917015  1.06284272  0.08967678  0.85809784
#>  [6]  0.92250793 -0.42511977  1.02580855  0.65080175 -0.26678263
#> [11]  0.56790313  1.07273123  1.02725011  0.97315528  0.96767992
#> [16]  0.48353934  0.86209519
```

We also need a penalty term. Here, it's the effective number of parameters (p\_waic). Each observation contributes to the penalty term, using the variance of its log-likelihoods.

``` r
# effective number of parameters by taking variance of log-likelihood of each
# observation 
each_p_waic <- log_lik(m4) %>% apply(2, var)
each_p_waic
#>  [1] 0.20005687 0.06593731 0.04132520 0.16133253 0.11968305 0.10756075
#>  [7] 0.70470686 0.05237387 0.13511036 0.36832509 0.12762613 0.03882063
#> [13] 0.04876040 0.07018034 0.06361188 0.25044359 0.09896701
```

The WAIC is the difference in total lppd and total p\_waic on the deviance scale.

``` r
lppd <- sum(each_lppd)
lppd
#> [1] 11.49707
p_waic <- sum(each_p_waic)
p_waic
#> [1] 2.654822
waic <- -2 * (lppd - p_waic)
waic
#> [1] -17.6845

# skip the summing step
each_waic <- -2 * (each_lppd - each_p_waic)
sum(each_waic)
#> [1] -17.6845
```

But because each point contributes information to the lppd and p\_waic calculations, we can compute a standard error on these numbers.

``` r
# standard error of the waics
se <- function(xs) sqrt(length(xs) * var(xs))
se(each_p_waic)
#> [1] 0.6824496
se(each_waic)
#> [1] 5.037537
```

These calculations match the estimates from the loo package. The `loo::waic` estimates use expected lppd (elpd) which already subtracts p\_waic from lppd. It's *expected* because it is the log-likelihood of the in-sample data adjusted with a penalty term. The resulting value is an expectation for out-of-sample data.

``` r
# manual elpd_waic
lppd - p_waic
#> [1] 8.842249
se(each_lppd - each_p_waic)
#> [1] 2.518768

loo::waic(m4)
#> Computed from 4000 by 17 log-likelihood matrix
#> 
#>           Estimate  SE
#> elpd_waic      8.8 2.5
#> p_waic         2.7 0.7
#> waic         -17.7 5.0
#> Warning: 1 (5.9%) p_waic estimates greater than 0.4.
#> We recommend trying loo() instead.
```

Getting WAIC for the models
---------------------------

Get the WAIC using the loo package.

I'm going to experiment with the [many-models](http://r4ds.had.co.nz/many-models.html)/data-frame-of-models technique here. That means doing some as-of-2016 unconventional things with data-frames.

Basically, I make a data-frame with one row per model and then do things to each model, storing the results of those operations as new columns in the data-frame. Specifically, I calculate the WAIC of the Stan models and refit the models using the classical technique so I can get an AIC value to compare to the WAIC value. Then I will compute Akaike weights for based on ELPD values.

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

model_summary
#> # A tibble: 4 x 11
#>         StanFit                           Formula ClassicalFit  Deviance
#>          <list>                             <chr>       <list>     <dbl>
#> 1 <S3: stanreg>                    kcal.per.g ~ 1    <S3: glm> -12.45830
#> 2 <S3: stanreg>            kcal.per.g ~ neocortex    <S3: glm> -12.87418
#> 3 <S3: stanreg>             kcal.per.g ~ log_mass    <S3: glm> -14.73810
#> 4 <S3: stanreg> kcal.per.g ~ neocortex + log_mass    <S3: glm> -25.35542
#> # ... with 7 more variables: AIC <dbl>, waic <dbl>, se_waic <dbl>,
#> #   elpd_waic <dbl>, se_elpd_waic <dbl>, p_waic <dbl>, se_p_waic <dbl>
```

The `rethinking::compare` function computes differences in WAIC and model weights automatically, but won't work with RStanARM models so we compute those by hand.

``` r
model_summary <- model_summary %>% 
  mutate(diff_waic = min(waic) - waic,
         weight = exp(elpd_waic) / sum(exp(elpd_waic)))
```

We obtain the following table summarizing the models:

``` r
# We need to exclude the list columns in order to print a formatted table 
model_summary %>% 
  select(-StanFit, -ClassicalFit) %>% 
  arrange(desc(weight)) %>% 
  knitr::kable(digits = 2) 
```

| Formula                            |  Deviance|     AIC|    waic|  se\_waic|  elpd\_waic|  se\_elpd\_waic|  p\_waic|  se\_p\_waic|  diff\_waic|  weight|
|:-----------------------------------|---------:|-------:|-------:|---------:|-----------:|---------------:|--------:|------------:|-----------:|-------:|
| kcal.per.g ~ neocortex + log\_mass |    -25.36|  -17.36|  -17.68|      5.04|        8.84|            2.52|     2.65|         0.68|        0.00|    0.96|
| kcal.per.g ~ log\_mass             |    -14.74|   -8.74|   -9.48|      4.46|        4.74|            2.23|     1.90|         0.44|       -8.21|    0.02|
| kcal.per.g ~ 1                     |    -12.46|   -8.46|   -9.15|      4.10|        4.57|            2.05|     1.22|         0.30|       -8.54|    0.01|
| kcal.per.g ~ neocortex             |    -12.87|   -6.87|   -7.69|      3.63|        3.85|            1.81|     1.84|         0.32|       -9.99|    0.01|

Remember what the book says about these weights:

> But what do these weights mean? There actually isn't a consensus about that. But here's Akaike's interpretation, which is common.
>
> > A model's weight is an estimate of the probability that the model will make the best predictions on new data, conditional on the set of models considered.
>
> Here’s the heuristic explanation. First, regard WAIC as the expected deviance of a model on future data. That is to say that WAIC gives us an estimate of E(D\_test). Akaike weights convert these deviance values, which are log-likelihoods, to plain likelihoods and then standardize them all. This is just like Bayes’ theorem uses a sum in the denominator to standardize the product of the likelihood and prior. Therefore the Akaike weights are analogous to posterior probabilities of models, conditional on expected future data.

McElreath goes on to note "However, given all the strong assumptions about repeat sampling that go into calculating WAIC, you cannot take this heuristic too seriously." The authors of the loo package themselves [no longer provide weights](https://github.com/stan-dev/loo/releases/tag/v0.1.5), citing a different limitation.

> In previous versions of **loo** model weights were also reported by `compare`. We have removed the weights because they were based only on the point estimate of the elpd values ignoring the uncertainty.
