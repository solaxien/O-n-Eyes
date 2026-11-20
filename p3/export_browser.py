"""Export the adapted checkpoint as independently loadable browser components.

FP16 neural layers, FP32 outputs/scheduler, batch-one guidance and external
weight shards limit simultaneous allocations. This is a new measured route;
the native and CUDA scores do not establish browser correctness.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx.external_data_helper import set_external_data
import torch
from diffusers import AutoencoderTiny, DPMSolverMultistepScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer

from p3.models import CHECKPOINTS, TAESD, resolve_model, resolve_taesd
from p3.provenance import file_hash, source_hashes, write_json, establish_protocol, exclusive_run


class Text(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids.to(torch.int64), return_dict=False)[0].float()


class Denoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.model(sample, timestep, encoder_hidden_states, return_dict=False)[0].float()


class Decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latent):
        return self.model.decode(latent, return_dict=False)[0].float()


def shard(graph_path, maximum=64*1024*1024):
    model = onnx.load(graph_path)
    files, stream, offset = [], None, 0
    for tensor in model.graph.initializer:
        if len(tensor.raw_data) < 1024:
            continue
        size = len(tensor.raw_data)
        if stream is None or offset + size > maximum:
            if stream:
                stream.close()
            name = f'{graph_path.stem}-{len(files):02d}.bin'
            files.append(name)
            stream = (graph_path.parent/name).open('wb')
            offset = 0
        stream.write(tensor.raw_data)
        set_external_data(tensor, location=files[-1], offset=offset, length=size)
        tensor.ClearField('raw_data')
        offset += size
    if stream:
        stream.close()
    onnx.save_model(model, graph_path)
    onnx.checker.check_model(str(graph_path))
    return model, files


def export_component(name, module, inputs, names, out, dynamic=None):
    path = out/f'{name}.onnx'
    module.eval()
    with torch.inference_mode():
        expected = module(*inputs).cpu().numpy().astype('<f4')
        torch.onnx.export(module, inputs, str(path), input_names=names, output_names=['output'],
                          opset_version=17, dynamo=False, do_constant_folding=True, dynamic_axes=dynamic)
    model, externals = shard(path)
    reference = {'inputs':{},'output':{'path':f'{name}-expected.f32','shape':list(expected.shape)}}
    expected.tofile(out/reference['output']['path'])
    for key, value in zip(names, inputs):
        array = value.cpu().numpy()
        filename = f'{name}-{key}.tensor'
        array.tofile(out/filename)
        reference['inputs'][key] = {'path':filename,'shape':list(array.shape),'dtype':str(array.dtype)}
    entries = []
    for filename in [path.name, *externals]:
        file = out/filename
        entries.append({'path':filename,'bytes':file.stat().st_size,'sha256':file_hash(file)})
    proof = {'graph':path.name,'files':entries,'operators':sorted(set(n.op_type for n in model.graph.node)),
             'reference':reference,'precision':'fp16 weights and neural inputs; fp32 output'}
    write_json(out/f'{name}.json', proof)
    print(json.dumps({'component':name,'bytes':sum(f['bytes'] for f in entries),'operators':proof['operators']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('.models/browser/v1'))
    parser.add_argument('--component',choices=['text','unet','decoder','all'],default='all')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--text-precision',choices=['fp16','mixed-fp32'],default='fp16')
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with exclusive_run(args.out):
        execute(args)


def execute(args):
    model_dir, _ = resolve_model('tiny','fp16',include_safety=False)
    vae_dir, _ = resolve_taesd()
    torch.set_num_threads(4)
    dtype, device = torch.float16,args.device
    adapter = Path('p3/weights/t38-coco.safetensors').resolve()
    establish_protocol(args.out,{'schema':1,'source':source_hashes('export_browser.py'),
        'model':CHECKPOINTS['tiny'].__dict__,'decoder':TAESD.__dict__,'adapter':file_hash(adapter),
        'text_precision':args.text_precision,'device':device,'torch':torch.__version__,'onnx':onnx.__version__})
    tokenizer = CLIPTokenizer.from_pretrained(model_dir/'tokenizer',local_files_only=True)
    ids = tokenizer('A red apple beside a blue cup.',padding='max_length',max_length=77,return_tensors='pt').input_ids
    generator = torch.Generator().manual_seed(38)
    sample = torch.randn(1,4,32,32,generator=generator).to(device=device,dtype=dtype)
    context = torch.randn(1,77,768,generator=generator).to(device=device,dtype=dtype)
    components = ['text','decoder','unet'] if args.component=='all' else [args.component]
    for part in components:
        if (args.out/f'{part}.json').exists():
            proof=json.loads((args.out/f'{part}.json').read_text())
            if any(not (args.out/f['path']).is_file() or file_hash(args.out/f['path'])!=f['sha256'] for f in proof['files']):
                raise ValueError('A completed export has changed bytes. Preserve it and choose a new output directory.')
            print(f'{part}: existing completed export retained',flush=True)
            continue
        if part=='text':
            model = CLIPTextModel.from_pretrained(model_dir/'text_encoder',variant='fp16',
                local_files_only=True,torch_dtype=dtype if args.text_precision=='fp16' else torch.float32,attn_implementation='eager').to(device)
            # The production text model uses local WASM: GPU text had a measured
            # late-token discrepancy even in FP32. The optional precision study
            # keeps embeddings FP16 to fit the mobile buffer-binding limit.
            if args.text_precision=='mixed-fp32':
                model.text_model.embeddings.token_embedding.weight.data = model.text_model.embeddings.token_embedding.weight.data.half()
            export_component(part,Text(model),(ids.to(device=device,dtype=torch.int32),),['input_ids'],args.out)
        elif part=='decoder':
            model = AutoencoderTiny.from_pretrained(vae_dir,local_files_only=True,torch_dtype=dtype).to(device)
            export_component(part,Decoder(model),(sample,),['latent'],args.out,
                {'latent':{2:'height',3:'width'},'output':{2:'image_height',3:'image_width'}})
        else:
            model = UNet2DConditionModel.from_pretrained(model_dir/'unet',variant='fp16',
                local_files_only=True,torch_dtype=dtype).to(device)
            model.load_lora_adapter(str(adapter.parent),weight_name=adapter.name,prefix='unet',local_files_only=True)
            model.fuse_lora(safe_fusing=True)
            model.unload_lora()
            model.set_attn_processor(AttnProcessor())
            export_component(part,Denoiser(model),(sample,torch.tensor([999.0],device=device),context),
                ['sample','timestep','encoder_hidden_states'],args.out,
                {'sample':{2:'height',3:'width'},'output':{2:'height',3:'width'}})
        del model
        gc.collect()
        if device=='cuda':
            torch.cuda.empty_cache()
    for name in ['vocab.json','merges.txt','tokenizer_config.json']:
        shutil.copyfile(model_dir/'tokenizer'/name,args.out/name)
    tests = ['','A red apple beside a blue cup.','Two yellow bananas, left of one apple.','A café with three chairs.']
    write_json(args.out/'tokenizer-tests.json',[{'text':t,'ids':tokenizer(t,padding='max_length',max_length=77).input_ids} for t in tests])
    schedules={}
    for steps in [4,6,8,12,20]:
        scheduler=DPMSolverMultistepScheduler.from_pretrained(model_dir,subfolder='scheduler',
            algorithm_type='dpmsolver++',solver_order=2,local_files_only=True)
        scheduler.set_timesteps(steps)
        schedules[str(steps)]={'timesteps':scheduler.timesteps.tolist(),'sigmas':scheduler.sigmas.tolist()}
    write_json(args.out/'schedules.json',schedules)
    manifest={'schema':1,'model':CHECKPOINTS['tiny'].__dict__,'decoder':TAESD.__dict__,
        'adapter_sha256':file_hash(adapter),'source':source_hashes('export_browser.py'),
        'components':{p:json.loads((args.out/f'{p}.json').read_text()) for p in ['text','unet','decoder'] if (args.out/f'{p}.json').exists()},
        'scheduler':'DPM-Solver++ 2M, midpoint, final sigma zero','guidance':7.5,'text_backend':'wasm','text_precision':args.text_precision,
        'resolution':[256,384],'safety_checker':False,'execution':'browser only; no inference endpoint'}
    write_json(args.out/'manifest.json',manifest)


if __name__=='__main__':
    main()
