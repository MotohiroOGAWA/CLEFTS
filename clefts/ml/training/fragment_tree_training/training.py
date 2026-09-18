"""Schema-v5 action training entry point, with one batched teacher forward."""
from __future__ import annotations
import argparse
import json
import math
import tempfile
from pathlib import Path
import torch
from clefts.ml.input.source_action_structure import SourceActionStructure
from clefts.ml.specgen.spectrum_generator import create_spectrum_generator
from .model import ActionFragmentTreeTrainingModel


def validation_subset(pieces, fraction: float, seed: int):
    """Select a fixed, uniformly sampled subset of measurement records."""
    import random
    from ...input.action_batching import select_samples
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('validation_fraction must be greater than 0 and at most 1')
    total = sum(piece.num_samples for piece in pieces)
    if not total:
        raise ValueError('Validation dataset contains no samples')
    indices = sorted(random.Random(seed).sample(range(total), max(1, math.ceil(total * fraction))))
    selected = []
    offset = cursor = 0
    for piece in pieces:
        end = offset + piece.num_samples
        start = cursor
        while cursor < len(indices) and indices[cursor] < end:
            cursor += 1
        if cursor > start:
            selected.append(select_samples(piece, [index - offset for index in indices[start:cursor]]))
        offset = end
    return selected, indices


def train_actions(*, model_config: dict, train_dir: str | Path, val_dir: str | Path,
                  output_dir: str | Path, epochs: int = 1, batch_size: int = 4,
                  lr: float = 1e-4, device: str = "cuda", resume: str | Path | None = None,
                  weight_decay: float = 0.01, gradient_clip: float = 1.0,
                  absolute_weight: float = 1.0, next_weight: float = 1.0,
                  negative_weight: float = 1.0, intensity_weight: float = 1.0,
                  absolute_intensity_weight: float = 1.0, max_samples: int = 128,
                  seed: int = 42, warmup_steps: int = 100, lr_patience: int = 3,
                  early_stopping_patience: int = 10, min_lr: float = 1e-6,
                  validation_interval_steps: int = 0, validation_fraction: float = 0.1,
                  train_mol_encoder: bool = False,
                  initialize_from: str | Path | None = None) -> dict:
    from dataclasses import replace
    import random
    import time
    from .sources import inherit_model_config
    from ...input.action_batching import select_samples
    from ...specgen.post_materialization_model import deduplicate_molecular_graphs
    from torch.utils.tensorboard import SummaryWriter
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    settings={key:value for key,value in locals().copy().items() if key in (
        'train_dir','val_dir','epochs','batch_size','lr','device','resume','weight_decay','gradient_clip',
        'absolute_weight','next_weight','negative_weight','intensity_weight','absolute_intensity_weight',
        'max_samples','seed','warmup_steps','lr_patience','early_stopping_patience','min_lr','train_mol_encoder','initialize_from',
        'validation_interval_steps','validation_fraction')}
    settings={key:str(value) if isinstance(value,Path) else value for key,value in settings.items()}
    (output/'training_args.json').write_text(json.dumps(settings,indent=2))
    if epochs<1 or batch_size<1 or max_samples<1 or not math.isfinite(lr) or lr<=0:
        raise ValueError('epochs, batch_size, max_samples and lr must be positive')
    if any(not math.isfinite(value) or value<0 for value in (weight_decay,gradient_clip,absolute_weight,next_weight,negative_weight,intensity_weight,absolute_intensity_weight,min_lr)):
        raise ValueError('Optimizer and loss settings must be finite and non-negative')
    if absolute_weight+next_weight+intensity_weight<=0:raise ValueError('Enable at least one training loss')
    if min_lr>lr or min(warmup_steps,lr_patience,early_stopping_patience)<0:
        raise ValueError('min_lr must not exceed lr; schedule patience/steps must be non-negative')
    if not isinstance(validation_interval_steps, int) or validation_interval_steps < 0:
        raise ValueError('validation_interval_steps must be a non-negative integer')
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction <= 1:
        raise ValueError('validation_fraction must be greater than 0 and at most 1')
    if initialize_from is not None and resume is not None:raise ValueError('Use initialize_from or resume, not both.')
    device=torch.device(device)
    if device.type!='cuda' or not torch.cuda.is_available():
        raise ValueError('Fragment Tree Training requires CUDA; CPU training is not supported')
    random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    checkpoint=torch.load(resume,map_location='cpu',weights_only=False) if resume else None
    prior_settings=checkpoint.get('training_settings',{}) if checkpoint else {}
    subset_seed=prior_settings.get('validation_subset_seed',prior_settings.get('seed',seed))
    settings['validation_subset_seed']=subset_seed
    if checkpoint:
        if model_config.get('fine_tuning') and model_config.get('fine_tuning')!=checkpoint['model_config'].get('fine_tuning'):raise ValueError('Resume fine-tuning configuration differs from checkpoint')
        model_config=checkpoint['model_config']
        prior=checkpoint.get('training_settings',{})
        if not prior:settings['train_mol_encoder']=not bool(model_config.get('fine_tuning'))
        # Restoring optimizer and loss definitions makes a resumed curve comparable.
        for key in ('absolute_weight','next_weight','negative_weight','intensity_weight','absolute_intensity_weight','train_mol_encoder','gradient_clip','warmup_steps','lr_patience','min_lr','lr'):
            if key in prior:settings[key]=prior[key]
        absolute_weight=settings['absolute_weight'];next_weight=settings['next_weight'];negative_weight=settings['negative_weight']
        intensity_weight=settings['intensity_weight'];absolute_intensity_weight=settings['absolute_intensity_weight'];train_mol_encoder=settings['train_mol_encoder']
        gradient_clip=settings['gradient_clip'];warmup_steps=settings['warmup_steps'];lr_patience=settings['lr_patience'];min_lr=settings['min_lr'];lr=settings['lr']
    params=model_config.get('params',model_config)
    model_config=inherit_model_config(model_config,train_dir,val_dir,encoder_checkpoint=params.get('mol_encoder_checkpoint'),saved_model=model_config if resume or params.get('fine_tuning') else None)
    model_config['max_samples']=max_samples
    (output/'training_config.json').write_text(json.dumps({'model_config':model_config,'training':settings},indent=2))
    params=dict(model_config);encoder_path=params.pop('mol_encoder_checkpoint',None)
    generator=create_spectrum_generator(params).to(device)
    if resume is None and initialize_from is None and not params.get('fine_tuning'):
        pretrained=torch.load(encoder_path,map_location='cpu',weights_only=False)
        generator.mol_encoder.load_state_dict(pretrained['mol_encoder_state_dict'])
    if not train_mol_encoder:generator.feature_model.freeze_mol_encoder()
    model=ActionFragmentTreeTrainingModel(generator.feature_model,downstream_model=generator.post_model,
        absolute_weight=absolute_weight,next_weight=next_weight,negative_weight=negative_weight,intensity_weight=intensity_weight,absolute_intensity_weight=absolute_intensity_weight).to(device)
    model.set_checkpoint_model_config(model_config)
    legacy=checkpoint is not None and not checkpoint.get('training_settings')
    optimizer_parameters=list(model.parameters()) if legacy else [p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(optimizer_parameters,lr=lr,weight_decay=weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=lr_patience,min_lr=min_lr)
    start=0;global_step=0;best=float('inf');bad_epochs=0;history=[];intermediate_history=[]
    if checkpoint:
        if checkpoint.get('fragmentation_schema')!=generator.architecture:raise ValueError('Incompatible checkpoint architecture')
        model.load_state_dict(checkpoint['model_state_dict']);optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start=int(checkpoint['epoch']);global_step=int(checkpoint.get('global_step',0));best=float(checkpoint.get('best_validation_loss',best))
        history=checkpoint.get('history',[])
        intermediate_history=checkpoint.get('intermediate_validation_history',[])
        if not history and (Path(resume).parent/'metrics.json').is_file():history=json.loads((Path(resume).parent/'metrics.json').read_text()).get('history',[])
        bad_epochs=int(checkpoint.get('bad_epochs',0))
        if checkpoint.get('scheduler_state_dict'):scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('torch_rng_state') is not None:torch.set_rng_state(checkpoint['torch_rng_state'])
        if checkpoint.get('cuda_rng_state') is not None:torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
        if checkpoint.get('python_rng_state') is not None:random.setstate(checkpoint['python_rng_state'])
    elif initialize_from:
        if params.get('fine_tuning'):raise ValueError('Weight initialization cannot be combined with frozen-base expansion')
        weights=torch.load(initialize_from,map_location='cpu',weights_only=False)
        model.load_state_dict(weights['model_state_dict'])
    elif params.get('fine_tuning'):
        from .fine_tuning import initialize_from_base
        initialize_from_base(model,params)
    # Chemistry ends here; CPU tensor collation can continue per batch.
    # Forward/backward, validation and compatibility masks use CUDA tensors.
    datasets=[];dataset_report={}
    for name,directory in (('train',train_dir),('validation',val_dir)):
        files=sorted(Path(directory).rglob('*.preft.pt'))
        if not files:raise ValueError(f'No schema-v5 .preft.pt files in {directory}')
        pieces=[];samples=0;states=0;formulas=0
        for file in files:
            data=SourceActionStructure.load(file,max_action_count=generator.feature_model.max_action_count)
            if data.num_samples<1:raise ValueError(f"Structure contains no samples: {file}")
            if data.downstream is None:raise ValueError('Full training requires stored target molecular graphs')
            if data.precursor_next_index.numel()==0 and data.action_type.numel() and data.precursor_row_action_ptr.numel()==1:
                raise ValueError('Regenerate structures with recorded precursor alternatives')
            samples+=data.num_samples;states+=data.teacher_state_sample_index.numel();formulas+=data.downstream.formula_tensor.shape[0]
            pieces.extend(select_samples(data,range(i,min(i+max_samples,data.num_samples))) for i in range(0,data.num_samples,max_samples))
        datasets.append(pieces)
        dataset_report[name]={'files':len(files),'samples':samples,'teacher_states':states,'formula_targets':formulas,'chunks':len(pieces)}
    (output/'dataset_summary.json').write_text(json.dumps(dataset_report,indent=2))
    intermediate_pieces=[]
    if validation_interval_steps:
        intermediate_pieces,subset_indices=validation_subset(datasets[1],validation_fraction,subset_seed)
        (output/'intermediate_validation_subset.json').write_text(json.dumps({
            'seed':subset_seed,'fraction':validation_fraction,'total_samples':dataset_report['validation']['samples'],
            'sample_indices':subset_indices},indent=2))
    parameters={'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'frozen':sum(p.numel() for p in model.parameters() if not p.requires_grad)}
    writer=SummaryWriter(str(output/'tensorboard'),purge_step=start+1 if resume else None)
    iteration_writer=SummaryWriter(str(output/'tensorboard'/'iterations'),purge_step=global_step+1 if resume else None)
    writer.add_text('configuration/model',json.dumps(model_config,indent=2));writer.add_text('configuration/training',json.dumps(settings,indent=2))
    started=time.perf_counter();stop_reason='epochs_completed'
    def batches(pieces,shuffle):
        order=list(range(len(pieces)))
        if shuffle:random.shuffle(order)
        group=[];count=0
        for index in order:
            piece=pieces[index]
            if group and (len(group)>=batch_size or count+piece.num_samples>max_samples):
                yield SourceActionStructure.from_structures(group);group=[];count=0
            group.append(piece);count+=piece.num_samples
        if group:yield SourceActionStructure.from_structures(group)
    def metric_values(result):
        return {'loss':result.loss,'absolute_action_loss':result.absolute_loss,
                'next_action_loss':result.next_action_loss,'intensity_loss':result.downstream_output.loss,**result.metrics}
    def intermediate_validation(epoch):
        was_training=model.training
        totals={};sample_count=batch_count=0
        validation_started=time.perf_counter()
        try:
            model.eval()
            with torch.no_grad():
                for data in batches(intermediate_pieces,False):
                    data=replace(data,downstream=deduplicate_molecular_graphs(data.downstream,data.source_smiles)).to(device)
                    result=model(data)
                    for key,value in metric_values(result).items():
                        scalar=float(value.detach())
                        if not math.isfinite(scalar):raise FloatingPointError(f'Non-finite intermediate validation metric {key}')
                        totals[key]=totals.get(key,0.)+scalar*data.num_samples
                    sample_count+=data.num_samples;batch_count+=1
                    del result,data,value
        finally:
            model.train(was_training)
        row={'epoch':epoch,'global_step':global_step,'validation_samples':sample_count,
             'total_validation_samples':dataset_report['validation']['samples'],
             'validation_fraction':validation_fraction,'validation_batches':batch_count,
             'validation_seconds':time.perf_counter()-validation_started,
             'validation_loss':totals.pop('loss')/sample_count}
        row.update({'validation/'+key:value/sample_count for key,value in totals.items()})
        intermediate_history.append(row)
        (output/'intermediate_validation.json').write_text(json.dumps({'history':intermediate_history},indent=2))
        headers=list(dict.fromkeys(key for item in intermediate_history for key in item))
        (output/'intermediate_validation.tsv').write_text('\t'.join(headers)+'\n'+'\n'.join(
            '\t'.join(str(item.get(key,'')) for key in headers) for item in intermediate_history)+'\n')
        for key,value in row.items():
            if key not in ('epoch','global_step'):
                iteration_writer.add_scalar('intermediate_validation/'+key,value,global_step)
        iteration_writer.flush()
        print(json.dumps({'event':'intermediate_validation_end',**row}),flush=True)
    try:
        for epoch in range(start,start+epochs):
            epoch_start=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
            row={'epoch':epoch+1};gradient_values=[]
            for name,pieces in zip(('train','validation'),datasets):
                model.train(name=='train');totals={};sample_count=0;batch_count=0
                with torch.set_grad_enabled(name=='train'):
                    for data in batches(pieces,name=='train'):
                        # Repeated molecular graphs (including Source roots) are
                        # encoded once and gathered for each sample's tree.
                        source_keys=data.source_smiles
                        data=replace(data,downstream=deduplicate_molecular_graphs(data.downstream,source_keys)).to(device)
                        if name=='train':
                            if global_step<warmup_steps:
                                for group in optimizer.param_groups:group['lr']=lr*(global_step+1)/max(warmup_steps,1)
                            optimizer.zero_grad(set_to_none=True)
                        result=model(data)
                        if not torch.isfinite(result.loss):raise FloatingPointError(f'Non-finite {name} loss at epoch {epoch+1}')
                        if name=='train':
                            result.loss.backward()
                            norm=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),gradient_clip if gradient_clip>0 else float('inf'),error_if_nonfinite=True)
                            gradient_values.append(float(norm));optimizer.step();global_step+=1
                            iteration_writer.add_scalar('batch/train_loss',float(result.loss.detach()),global_step)
                            iteration_writer.add_scalar('batch/gradient_norm',float(norm),global_step)
                            iteration_writer.add_scalar('batch/learning_rate',optimizer.param_groups[0]['lr'],global_step)
                        values=metric_values(result)
                        for key,value in values.items():
                            scalar=float(value.detach())
                            if not math.isfinite(scalar):raise FloatingPointError(f'Non-finite metric {name}/{key}')
                            totals[key]=totals.get(key,0.)+scalar*data.num_samples
                        sample_count+=data.num_samples;batch_count+=1
                        # Release the training graph before evaluating a subset.
                        del values,result,data,value
                        if name=='train' and validation_interval_steps and global_step%validation_interval_steps==0:
                            intermediate_validation(epoch+1)
                row[name+'_loss']=totals.pop('loss')/max(sample_count,1)
                row.update({name+'/'+key:value/max(sample_count,1) for key,value in totals.items()})
                row[name+'/samples']=sample_count;row[name+'/batches']=batch_count
            row['global_step']=global_step;row['learning_rate']=optimizer.param_groups[0]['lr']
            row['gradient_norm']=sum(gradient_values)/max(len(gradient_values),1)
            row['epoch_seconds']=time.perf_counter()-epoch_start;row['cuda_peak_memory_mb']=torch.cuda.max_memory_allocated(device)/2**20
            improved=row['validation_loss']<best-1e-6
            if improved:best=row['validation_loss'];bad_epochs=0
            else:bad_epochs+=1
            if global_step>=warmup_steps:scheduler.step(row['validation_loss'])
            history.append(row)
            for key,value in row.items():
                if key!='epoch':writer.add_scalar(key,value,epoch+1)
            writer.flush();iteration_writer.flush()
            report={'history':history,'intermediate_validation_history':intermediate_history,'schema_version':5,'fragmentation_schema':generator.architecture,'model_config':model_config,'training':settings,'datasets':dataset_report,'parameters':parameters,'best_validation_loss':best,
                'elapsed_seconds':time.perf_counter()-started,'validation_mode':'teacher forcing on stored graphs; candidate recall measured before forced positives','stop_reason':'running'}
            (output/'metrics.json').write_text(json.dumps(report,indent=2))
            headers=list(dict.fromkeys(key for item in history for key in item));(output/'metrics.tsv').write_text('\t'.join(key.replace('/','_') for key in headers)+'\n'+'\n'.join('\t'.join(str(item.get(key,'')) for key in headers) for item in history)+'\n')
            saved=dict(model_config=model_config,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),epoch=epoch+1,global_step=global_step,intermediate_validation_history=intermediate_history,schema_version=5,fragmentation_schema=generator.architecture,history=history,training_settings=settings,best_validation_loss=best,bad_epochs=bad_epochs,torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),python_rng_state=random.getstate())
            torch.save(saved,output/'last.pt')
            if improved:torch.save(saved,output/'best.pt')
            print(json.dumps({'event':'epoch_end','current':epoch-start+1,'total':epochs,**row}),flush=True)
            if early_stopping_patience and bad_epochs>=early_stopping_patience:
                stop_reason='early_stopping';break
        report['stop_reason']=stop_reason
        (output/'metrics.json').write_text(json.dumps(report,indent=2));(output/'training_report.json').write_text(json.dumps(report,indent=2))
        lines=['# Fragment Tree Training Report','',f"Device: {device}; trainable parameters: {parameters['trainable']:,}",f"Best validation loss: {best:.6g}",f"Stop reason: {stop_reason}",'','Validation uses teacher forcing and stored molecular graphs. Filter recall is evaluated before positive insertion; this is not a free-running spectrum benchmark.','','| Epoch | Train loss | Validation loss | Filter intensity recall | Spectrum cosine | LR |','|---|---|---|---|---|---|']
        for item in history:lines.append(f"| {item['epoch']} | {item['train_loss']:.6g} | {item['validation_loss']:.6g} | {item.get('validation/intensity_recall_at_filter',0):.4f} | {item.get('validation/spectrum_cosine_similarity',0):.4f} | {item.get('learning_rate',0):.3g} |")
        if intermediate_history:
            lines.extend(['', '## Intermediate validation', '', 'These fixed-subset checks are observational; scheduling and best-checkpoint selection use full epoch validation.', '', '| Step | Epoch | Samples | Validation loss | Spectrum cosine |', '|---|---|---|---|---|'])
            for item in intermediate_history:
                lines.append(f"| {item['global_step']} | {item['epoch']} | {item['validation_samples']} | {item['validation_loss']:.6g} | {item.get('validation/spectrum_cosine_similarity',0):.4f} |")
        (output/'training_report.md').write_text('\n'.join(lines)+'\n')
        return report
    except Exception as error:
        (output/'training_failure.json').write_text(json.dumps({'error':str(error),'global_step':global_step,'completed_epochs':len(history)},indent=2))
        raise
    finally:
        writer.close();iteration_writer.close()


# Only trainable model sections are exposed by the training CLI.
MODEL_OPTIONS = {
    'action-hidden-dim': ('action_model_params', 'hidden_dim', int, 128),
    'action-condition-dim': ('action_model_params', 'condition_dim', int, 128),
    'action-num-heads': ('action_model_params', 'num_heads', int, 4),
    'action-max-roles': ('action_model_params', 'max_roles', int, None),
    'action-top-k': ('action_model_params', 'action_prefilter_top_k', int, 64),
    'action-max-k': ('action_model_params', 'action_prefilter_max_k', int, 128),
    'action-threshold': ('action_model_params', 'action_prefilter_threshold_logit', float, 1.0),
    'beam-size': ('action_model_params', 'beam_size', int, 32),
    'max-decode-steps': ('action_model_params', 'max_decode_steps', int, 16),
    'action-state-layers': ('action_model_params', 'state_num_layers', int, 2),
    'action-state-dropout': ('action_model_params', 'state_dropout', float, 0.0),
    'post-hidden-dim': ('post_model_params', 'hidden_dim', int, 128),
    'post-num-layers': ('post_model_params', 'num_layers', int, 2),
    'post-cosine-loss-weight': ('post_model_params', 'cosine_loss_weight', float, 0.5),
    'post-num-heads': ('post_model_params', 'num_heads', int, 4),
}


def training_model_config(args):
    config = {'max_samples':args.max_samples,'architecture': 'source-anchored-action-autoregressive-v1',
              'action_model_params': {}, 'post_model_params': {}}
    for flag, (section, key, _, default) in MODEL_OPTIONS.items():
        value = getattr(args, flag.replace('-', '_'))
        if value is not None or default is not None:
            config[section][key] = value if value is not None else default
    if args.mol_encoder_checkpoint:
        config['mol_encoder_checkpoint'] = args.mol_encoder_checkpoint
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mol-encoder-checkpoint', help='Pretrained Mol Encoder checkpoint; required for new training.')
    for flag, (_, _, kind, default) in MODEL_OPTIONS.items():
        parser.add_argument('--'+flag, type=kind, default=default)
    for name in ('train-dir','val-dir','output-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--epochs',type=int,default=1);parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--device',default='cuda');parser.add_argument('--lr',type=float,default=1e-4);parser.add_argument('--resume')
    parser.add_argument('--initialize-from',help='Initialize compatible pretrained weights with a fresh optimizer for fine-tuning.')
    parser.add_argument('--fine-tune-checkpoint',help='Base action training model.pt to expand with a new cleavage pattern set.')
    parser.add_argument('--adapter-width',type=int,default=8,help='Extra low-rank nodes per linear/attention projection (default: 8).')
    for name, default in (("weight-decay",0.01),("gradient-clip",1.0),
                          ("absolute-weight",1.0),("next-weight",1.0),
                          ("negative-weight",1.0),("intensity-weight",1.0),("absolute-intensity-weight",1.0),("min-lr",1e-6)):
        parser.add_argument('--'+name,type=float,default=default)
    for name,default in (('max-samples',128),('seed',42),('warmup-steps',100),('lr-patience',3),('early-stopping-patience',10)):
        parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--train-mol-encoder',action='store_true',help='Update the pretrained encoder; default is frozen.')
    parser.add_argument('--validation-interval-steps',type=int,default=0,
        help='Evaluate a validation subset every N optimizer steps; 0 disables intermediate validation.')
    parser.add_argument('--validation-fraction',type=float,default=0.1,
        help='Fraction of validation samples used for intermediate checks (0 < fraction <= 1; default: 0.1).')
    return parser


def main(argv: list[str] | None = None) -> None:
    args=build_arg_parser().parse_args(argv)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    (output/'training_args.json').write_text(json.dumps(vars(args),indent=2))
    if not (args.resume or args.fine_tune_checkpoint or args.mol_encoder_checkpoint):
        raise SystemExit('--mol-encoder-checkpoint is required for new training.')
    model_config=training_model_config(args)
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
                pattern_set_path=patterns_path,new_params_path=config_path,width=args.adapter_width)
            model_config['fine_tuning'].pop('pattern_set_path', None)
    report=train_actions(model_config=model_config,train_dir=args.train_dir,val_dir=args.val_dir,
        output_dir=args.output_dir,epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,device=args.device,resume=args.resume,
        weight_decay=args.weight_decay,gradient_clip=args.gradient_clip,
        absolute_weight=args.absolute_weight,next_weight=args.next_weight,
        negative_weight=args.negative_weight,intensity_weight=args.intensity_weight,absolute_intensity_weight=args.absolute_intensity_weight,max_samples=args.max_samples,
        seed=args.seed,warmup_steps=args.warmup_steps,lr_patience=args.lr_patience,early_stopping_patience=args.early_stopping_patience,min_lr=args.min_lr,train_mol_encoder=args.train_mol_encoder,initialize_from=args.initialize_from,
        validation_interval_steps=args.validation_interval_steps,validation_fraction=args.validation_fraction)
    print(json.dumps(report))

if __name__=='__main__':main()
