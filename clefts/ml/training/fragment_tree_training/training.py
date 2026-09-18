"""Schema-v4 action training entry point, with one batched teacher forward."""
from __future__ import annotations
import argparse
import json
import math
import tempfile
from clefts.ml.specgen.config_options import configure_model_options, resolve_model_options
from pathlib import Path
import torch
from clefts.ml.input.source_action_structure import SourceActionStructure
from clefts.ml.specgen.spectrum_generator import create_spectrum_generator
from .model import ActionFragmentTreeTrainingModel


def train_actions(*, model_config: dict, train_dir: str | Path, val_dir: str | Path,
                  output_dir: str | Path, epochs: int = 1, batch_size: int = 4,
                  lr: float = 1e-4, device: str = "cpu", resume: str | Path | None = None,
                  weight_decay: float = 0.01, gradient_clip: float = 0.0,
                  absolute_weight: float = 1.0, next_weight: float = 1.0,
                  negative_weight: float = 1.0, intensity_weight: float = 1.0,
                  initialize_from: str | Path | None = None) -> dict:
    if epochs < 1 or batch_size < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("epochs and batch_size must be positive; lr must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in
           (weight_decay, gradient_clip, absolute_weight, next_weight, negative_weight, intensity_weight)):
        raise ValueError("Optimizer settings and loss weights must be finite and non-negative")
    if initialize_from is not None and resume is not None:
        raise ValueError('Use initialize_from or resume, not both.')
    resume_checkpoint = None
    if resume is not None:
        resume_checkpoint = torch.load(resume, map_location=device, weights_only=False)
        if not resume_checkpoint.get("model_config"):
            raise ValueError("Resume checkpoint must contain model configuration")
        model_config = resume_checkpoint["model_config"]
    from .sources import inherit_model_config
    current_params = model_config.get("params", model_config)
    model_config = inherit_model_config(model_config, train_dir, val_dir,
        encoder_checkpoint=current_params.get("mol_encoder_checkpoint"),
        saved_model=model_config if resume is not None or current_params.get("fine_tuning") else None)
    params = dict(model_config)
    mol_encoder_checkpoint = params.pop("mol_encoder_checkpoint", None)
    # install_expansion (frozen base + low-rank adapters) already ran inside
    # the generator constructor when params['fine_tuning'] is set.
    generator=create_spectrum_generator(params).to(device)
    if getattr(generator,"architecture",None)!="source-anchored-action-autoregressive-v1":
        raise ValueError("Action training requires the Source/action architecture")
    if resume is None and initialize_from is None and not params.get("fine_tuning") and mol_encoder_checkpoint:
        encoder_checkpoint = torch.load(mol_encoder_checkpoint, map_location=device, weights_only=False)
        encoder_state = encoder_checkpoint.get("mol_encoder_state_dict")
        if encoder_state is None:
            raise ValueError("Mol encoder checkpoint must contain mol_encoder_state_dict")
        generator.mol_encoder.load_state_dict(encoder_state)
    model=ActionFragmentTreeTrainingModel(generator.feature_model,downstream_model=generator.post_model, absolute_weight=absolute_weight, next_weight=next_weight, negative_weight=negative_weight, intensity_weight=intensity_weight).to(device)
    model.set_checkpoint_model_config(model_config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    start=0
    if resume is not None:
        checkpoint=resume_checkpoint
        if checkpoint.get("fragmentation_schema")!=generator.architecture:
            raise ValueError("Cannot resume a checkpoint from a different architecture as an action model")
        model.load_state_dict(checkpoint['model_state_dict']);optimizer.load_state_dict(checkpoint['optimizer_state_dict']);start=int(checkpoint['epoch'])
    elif initialize_from is not None:
        if params.get('fine_tuning'): raise ValueError('Weight initialization cannot be combined with frozen-base expansion.')
        checkpoint=torch.load(initialize_from,map_location=device)
        if checkpoint.get('fragmentation_schema')!=generator.architecture:
            raise ValueError('Pretrained checkpoint has an incompatible fragmentation architecture')
        model.load_state_dict(checkpoint['model_state_dict'])
    elif params.get("fine_tuning"):
        # Fresh fine-tuning start: copy every frozen-base tensor once, before
        # any optimizer step touches the newly expanded rows/adapters.
        from .fine_tuning import initialize_from_base
        initialize_from_base(model,params)
    loaders=[]
    for directory in (train_dir,val_dir):
        files=sorted(Path(directory).rglob('*.preft.pt'))
        if not files: raise ValueError(f"No schema-v4 .preft.pt files in {directory}")
        # Loading explicitly rejects v3; there is no automatic conversion.
        structures=[SourceActionStructure.load(path) for path in files]
        if any(item.downstream is None for item in structures):
            raise ValueError("Full training requires saved target molecular graphs")
        loaders.append(torch.utils.data.DataLoader(structures,batch_size=batch_size,shuffle=directory==train_dir,collate_fn=SourceActionStructure.from_structures))
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    history=[]
    global_step=0
    metrics_path=output/"metrics.tsv"
    metric_headers=None
    for epoch in range(start,start+epochs):
        row={'epoch':epoch+1}
        for name,loader in zip(('train','validation'),loaders):
            model.train(name=='train')
            losses=[]
            totals={}
            with torch.set_grad_enabled(name=='train'):
                for data in loader:
                    result=model(data.to(device))
                    if name=='train':
                        optimizer.zero_grad(set_to_none=True);result.loss.backward()
                        if gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                        optimizer.step();global_step+=1
                    losses.append(float(result.loss.detach()))
                    totals.setdefault('absolute_action_loss',[]).append(float(result.absolute_loss.detach()))
                    totals.setdefault('next_action_loss',[]).append(float(result.next_action_loss.detach()))
                    downstream=result.downstream_output
                    if downstream is not None:
                        intensity_loss=downstream['loss'] if isinstance(downstream,dict) else downstream.loss
                        totals.setdefault('intensity_loss',[]).append(float(intensity_loss.detach()))
                    for metric,value in result.metrics.items(): totals.setdefault(metric,[]).append(float(value.detach()))
            row[name+'_loss']=sum(losses)/len(losses)
            row.update({name+'/'+metric:sum(values)/len(values) for metric,values in totals.items()})
        row['global_step']=global_step
        history.append(row)
        if metric_headers is None:
            metric_headers=list(row)
            metrics_path.write_text('\t'.join(key.replace('/', '_') for key in metric_headers)+'\n')
        with metrics_path.open('a') as metrics_file:
            metrics_file.write('\t'.join(str(row[key]) for key in metric_headers)+'\n')
        (output/'metrics.json').write_text(json.dumps({'history':history},indent=2))
        print(json.dumps({'event':'epoch_end','epoch':epoch+1,'current':epoch-start+1,'total':epochs,**row}),flush=True)
        torch.save(dict(model_config=model_config,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch+1,schema_version=4,fragmentation_schema=generator.architecture),output/'last.pt')
    report={'history':history,'schema_version':4,'fragmentation_schema':generator.architecture}
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__)
    configure_model_options(parser)
    for name in ('train-dir','val-dir','output-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--epochs',type=int,default=1);parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--device',default='cpu');parser.add_argument('--lr',type=float,default=1e-4);parser.add_argument('--resume')
    parser.add_argument('--initialize-from',help='Initialize compatible pretrained weights with a fresh optimizer for fine-tuning.')
    parser.add_argument('--fine-tune-checkpoint',help='Base action training model.pt to expand with a new cleavage pattern set.')
    parser.add_argument('--fine-tune-pattern-set',help='Complete new .clevageset.json including all old patterns.')
    parser.add_argument('--adapter-width',type=int,default=8,help='Extra low-rank nodes per linear/attention projection (default: 8).')
    for name, default in (("weight-decay",0.01),("gradient-clip",0.0),
                          ("absolute-weight",1.0),("next-weight",1.0),
                          ("negative-weight",1.0),("intensity-weight",1.0)):
        parser.add_argument('--'+name,type=float,default=default)
    return parser


def main(argv: list[str] | None = None) -> None:
    args=build_arg_parser().parse_args(argv)
    if args.fine_tune_pattern_set and not args.fine_tune_checkpoint:
        raise SystemExit('--fine-tune-pattern-set requires --fine-tune-checkpoint.')
    model_config=resolve_model_options(args)
    from .sources import inherit_model_config
    saved_model = None
    source_checkpoint = args.resume or args.fine_tune_checkpoint
    if source_checkpoint:
        saved_model = torch.load(source_checkpoint, map_location='cpu', weights_only=False)['model_config']
    # A new expansion inherits the encoder from its base, and Fragmenter from the expanded datasets.
    if args.fine_tune_checkpoint and not args.resume:
        saved_params = saved_model.get('params', saved_model)
        from .sources import dataset_sources
        inherited = dataset_sources(args.train_dir, args.val_dir)
        saved_model = {**saved_params, 'fragmenter_params': inherited['fragmenter_params'],
                       'adduct_type_strs': inherited['adduct_type_strs']}
    model_config = inherit_model_config(model_config, args.train_dir, args.val_dir,
        encoder_checkpoint=model_config.get('mol_encoder_checkpoint'), saved_model=saved_model)
    if args.fine_tune_checkpoint and not args.resume:
        from .fine_tuning import prepare_model_config
        with tempfile.TemporaryDirectory(prefix='clefts-training-config-') as directory:
            config_path=Path(directory)/'model.json'
            config_path.write_text(json.dumps(model_config))
            patterns_path=Path(directory)/'patterns.json'
            patterns_path.write_text(json.dumps(model_config['fragmenter_params']['fragment_ion_tree_builder']['cleavage_pattern_set']))
            model_config=prepare_model_config(checkpoint_path=args.fine_tune_checkpoint,
                pattern_set_path=args.fine_tune_pattern_set or patterns_path,new_params_path=config_path,width=args.adapter_width)
            if not args.fine_tune_pattern_set:
                model_config['fine_tuning'].pop('pattern_set_path', None)
    report=train_actions(model_config=model_config,train_dir=args.train_dir,val_dir=args.val_dir,
        output_dir=args.output_dir,epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,device=args.device,resume=args.resume,
        weight_decay=args.weight_decay,gradient_clip=args.gradient_clip,
        absolute_weight=args.absolute_weight,next_weight=args.next_weight,
        negative_weight=args.negative_weight,intensity_weight=args.intensity_weight,initialize_from=args.initialize_from)
    print(json.dumps(report))

if __name__=='__main__':main()
