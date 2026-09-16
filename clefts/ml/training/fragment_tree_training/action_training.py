"""Schema-v4 action training entry point, with one batched teacher forward."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from clefts.ml.input.source_action_structure import SourceActionStructure
from clefts.ml.specgen.fragment_tree_spectrum_predictor import create_spectrum_generator
from .action_model import ActionFragmentTreeTrainingModel


def train_actions(*, model_config: dict, train_dir: str | Path, val_dir: str | Path,
                  output_dir: str | Path, epochs: int = 1, batch_size: int = 4,
                  lr: float = 1e-4, device: str = "cpu", resume: str | Path | None = None) -> dict:
    generator=create_spectrum_generator(model_config.get("params",model_config)).to(device)
    if getattr(generator,"architecture",None)!="source-anchored-action-autoregressive-v1":
        raise ValueError("Action training requires the Source/action architecture")
    model=ActionFragmentTreeTrainingModel(generator.feature_model,downstream_model=generator.post_model).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    start=0
    if resume is not None:
        checkpoint=torch.load(resume,map_location=device)
        if checkpoint.get("fragmentation_schema")!=generator.architecture:
            raise ValueError("Cannot resume a legacy edge checkpoint as an action model")
        model.load_state_dict(checkpoint['model_state_dict']);optimizer.load_state_dict(checkpoint['optimizer_state_dict']);start=int(checkpoint['epoch'])
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
                        optimizer.zero_grad(set_to_none=True);result.loss.backward();optimizer.step()
                    losses.append(float(result.loss.detach()))
                    for metric,value in result.metrics.items(): totals.setdefault(metric,[]).append(float(value.detach()))
            row[name+'_loss']=sum(losses)/len(losses)
            row.update({name+'/'+metric:sum(values)/len(values) for metric,values in totals.items()})
        history.append(row)
        torch.save(dict(model_config=model_config,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            epoch=epoch+1,schema_version=4,fragmentation_schema=generator.architecture),output/'last.pt')
    report={'history':history,'schema_version':4,'fragmentation_schema':generator.architecture}
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('params','train-dir','val-dir','output-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--epochs',type=int,default=1);parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--device',default='cpu');parser.add_argument('--lr',type=float,default=1e-4);parser.add_argument('--resume')
    args=parser.parse_args()
    report=train_actions(model_config=json.loads(Path(args.params).read_text()),train_dir=args.train_dir,val_dir=args.val_dir,
        output_dir=args.output_dir,epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,device=args.device,resume=args.resume)
    print(json.dumps(report))

if __name__=='__main__':main()
