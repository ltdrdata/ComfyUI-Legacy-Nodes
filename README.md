# ComfyUI-Legacy-Nodes

This extension provides backward compatibility for the nodes prior to the commit mentioned for KSampler. The commit can be found at https://github.com/comfyanonymous/ComfyUI/commit/4eab00e14bc0b52a9c688486d7ee8b392e01020d.

## Nodes
* KSampler (Legacy)
- In the new approach, ComfyUI completely removed the GPU seed that was internally used by the SDE sampler. GPU seed and CPU seed generate completely different noise, so you will notice that previously generated prompts produce entirely different results on latest version. This node allows for the internal use of a seed, using the GPU seed, as in the previous method, to reproduce the previously generated results.

On the other hand, relying on the same GPU seed generally produces similar images, but you may observe subtle variations in the results each time it is run. This node is recommended to be used solely for the purpose of reproducing existing prompts.
