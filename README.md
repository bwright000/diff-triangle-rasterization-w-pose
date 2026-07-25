# Differentiable Triangle Rasterization (with pose optimisation)

Differentiable triangle rasteriser used by
**Triangle Splatting SLAM** ([code](https://github.com/nmjfry/ts-slam),
[project page](https://nmjfry.github.io/triangle-splatting-slam/)).

This is a fork of the [Mesh Splatting](https://github.com/meshsplatting/mesh-splatting)
triangle rasteriser, with added support for SLAM. The loss is back-propagated to
the **camera pose** (6-DoF) and to the **vertex world positions** and
**per-vertex opacity**, so the same rasteriser can be used for both tracking and
mapping. Triangle opacity is the **mean** of its three vertex weights.

If you use it in your own research, please cite the works below.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@misc{held2025triangle,
title={Triangle Splatting for Real-Time Radiance Field Rendering}, 
author={Jan Held and Renaud Vandeghen and Adrien Deliege and Abdullah Hamdi and Silvio Giancola and Anthony Cioppa and Andrea Vedaldi and Bernard Ghanem and Andrea Tagliasacchi and Marc Van Droogenbroeck},
year={2025},
eprint={2505.19175},
url={https://arxiv.org/abs/2505.19175}, 
}
</code></pre>
  </div>
</section>

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <pre><code>@inproceedings{Held20253DConvex,
title = {{3D} Convex Splatting: Radiance Field Rendering with {3D} Smooth Convexes},
author = {Jan Held and Renaud Vandeghen and Abdullah Hamdi and Adrien Deli\`ege and Anthony Cioppa and Silvio Giancola and Andrea Vedaldi and Bernard Ghanem and Marc Van Droogenbroeck},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
address = {Nashville, TN, USA},
month = {June},
year = {2025}
}
</code></pre>
  </div>
</section>

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <!-- <h2 class="title">BibTeX</h2> -->
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>