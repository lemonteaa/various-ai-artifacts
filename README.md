# Various AI Artifacts

Misc. collection of AI generated codes/docs...

Because sometimes it's hard to categorize (facepalm)

## List of artifacts

- **NTK Demo**

Neural Tangent Kernel + NNGP (Neural Network Gaussian Process)/Infinite Width limit + Direct Bayesian Inference to train NN, is an interesting but somewhat inaccessible topics. While there are software libraries that implemented it, I feel like those are "too academic" and difficult for practitioner (software engineers but not math people) to get started. So I thought, why not lookup the math formula manually and just ask AI to directly implement it in the most direct way possible? No abstraction, just straight numpy/scipy expressions.

(Note: have not been able to reproduce scaling up results from a quick test - probably a combination of using a more practical implementation (I read from a paper/blog that in practise the kernel matrix is roughly block diagonal), and changing up to use analytical kernel formula for an actual MLP rather than the toy example of single layer network, would be needed)

- **Inhouse LLM Pretraining script**

There is the huggingface transformer tutorial, but I personally found it to be quite tedious (long winded setups). AI basically one-shot the task successfully. (I did provide some carefully curated context, and it is allowed to use the transformer library - this is an infra task, not a test of pytorch tensor programming ability) Thought wouold share just in case anyone find it helpful.

- **GLM 4.7 Smoke test**

GLM 4.7 was just released (2025 Dec), so we test its general coding capability level. Mainly self contained tests that request a self contained webpage, vanilla html5 + css + js, CDN JS Library only. Usually visual tests (aka 3D demo) since those are the kinds where even layperson can "see" the results at a glance (like, literally see).

Two sub-entry: 1) Pagoda (Voxel 3D modelling), 2) Age of empire clone minigame (game engine + 3D)
