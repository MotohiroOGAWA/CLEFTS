"""Structure-level counts and mean per-sample assignment coverage for manifests."""
def structure_manifest_fields(structure):
    downstream=getattr(structure,'downstream',None)
    decoded=getattr(downstream,'decoded',None)
    annotations=getattr(structure,'sample_annotations',())
    def mean(key):
        values=[sample[key] for sample in annotations if sample.get(key) is not None]
        return sum(values)/len(values) if values else None
    return dict(num_nodes=sum(tree.num_nodes for tree in decoded.trees) if decoded else None,
                num_edges=sum(tree.num_edges for tree in decoded.trees) if decoded else None,
                assignment_score=mean('assignmentScore'),
                assignment_score_without_precursor=mean('assignmentScoreWithoutPrecursor'))
