

# The Bayesian Workflow

- based upon Gelman, Vehtari, Simpson 2020 (arxiv)

``` mermaid
flowchart TD
  init["Pick an initial model<br/><small>§2.1–2.3, 2.5 modular construction, scaling</small>"]
  ppc["Prior predictive check<br/><small>§2.4</small>"]
  fit["Fit the model<br/><small>§3 warmup, run length, approximate algorithms, fit fast fail fast</small>"]
  validate["Validate computation<br/><small>§4 fake-data simulation, simulation-based calibration</small>"]
  issues["Address computational issues<br/><small>§5 simplify, reparameterize, marginalize, add prior info or data</small>"]
  use["Evaluate and use model<br/><small>§6 posterior predictive check, cross validation, prior sensitivity, summarize</small>"]
  modify["Modify model<br/><small>§7 expand or replace components, add data, change priors</small>"]
  compare["Compare models<br/><small>§8 visualize side by side, cross validation, stacking/averaging</small>"]

  init --> ppc
  ppc -->|priors implausible| modify
  ppc --> fit
  fit --> validate
  validate -->|problems| issues
  issues --> fit
  validate -->|OK| use
  use --> modify
  modify -->|recheck| ppc
  use --> compare
```

## Tools

The whole thing: - Stan + R and its packages (trialr,…)

For model evaluation/prior predictive check/modification cycle - PyMC /
PyTensor - Pyro - Tensor Flow Probabilty (tensor friendly
probability?) - Numpyro - BlackJAX

emcee  - used a lot in astrophyiscs

Fit Model/adddress computational issues: - Bayeux +BlackJAX + JAXopt

For Validate Computation/Eval and use model/Compare Model - ArviZ

Others: ptax,

Colin - worked on PyMC, TF probability at Google/deep mind Bayeux maybe
uses Numpyro



Bayeux install
```sh
pip install bayeux_ml
```
