This repo is a work in progress. 

It is intended to contain functions extending `pytorch` and `torchnet` to add:
*	A set of distance Functions (Euclidean, Cosine, Poincare) which, given a pair of tensors A (m, d) and B (n, d), returns an (m, n) matrix of the distances between them.
* A pair of `torchnet` `Meter`s for Discounted Cumulative Gain and Normalized Discounted Cumulative Gain. 