# Semi-manual rigging tool

Idea: Rigging + Skinning + Animation re-targetting is a specialized skill for 3D artists that layman may find difficult to learn. While there are automatic tools to help, I'd also like something *local* to avoid vendor dependency. Since full automatic likely involve somes propietary AI model, or specialized algorithms that are niche and not open, my idea is to have a comprimose with a semi-manual tool.

Intended workflow:

1. Load GLB Mesh
2. Load BVH Skeleton
3. Drag Bones to fit Mesh
4. Bind & Export

The bone fitting part involve manual judgement for now, also remember to adjust influence radius (there is a next version planned with granular radius for "harder" model with mutli-scale situation).

Although overly large raidus may lead to mismatched bone to vertex, overly small radius currently lead to nonmatched vertex, which defaults to matching root bone, which leads to "Monstrous"/"exploding" mesh problem and is a show stopper.

Skinning/weighting bone-vertex match is something done automatically using a simple heuristic algorithm.

(Disclaimer: no promise made about the quality of results. In fact, probably expect poor results for any complex mesh.)

