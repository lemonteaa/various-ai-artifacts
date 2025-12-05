# Various AI Artifacts

Misc. collection of AI generated codes/docs...

Because sometimes it's hard to categorize (facepalm)

## List of artifacts

- **NTK Demo**

Neural Tangent Kernel + NNGP (Neural Network Gaussian Process)/Infinite Width limit + Direct Bayesian Inference to train NN, is an interesting but somewhat inaccessible topics. While there are software libraries that implemented it, I feel like those are "too academic" and difficult for practitioner (software engineers but not math people) to get started. So I thought, why not lookup the math formula manually and just ask AI to directly implement it in the most direct way possible? No abstraction, just straight numpy/scipy expressions.

(Note: have not been able to reproduce scaling up results from a quick test - probably a combination of using a more practical implementation (I read from a paper/blog that in practise the kernel matrix is roughly block diagonal), and changing up to use analytical kernel formula for an actual MLP rather than the toy example of single layer network, would be needed)
