import numpy as np
import torch
import importlib

from collections import defaultdict
from argparse import Namespace
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.fastmri import fft2c, ifft2c

def resolve_class(class_path: str):
    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not resolve class '{class_path}'. Error: {e}")

def test(model, data_loader, num_adj_slices):
    model['cnn'].eval()
    for task in ['brain', 'knee']:
        for acc in ['acc4', 'acc8']:
            model[task][acc].eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (mask, kspace, image, accs, fnames) in data_loader:
            mask = mask.cuda(non_blocking=True)
            kspace = kspace.cuda(non_blocking=True)
            image = image.cuda(non_blocking=True)
            acc = accs[0]
            fname = fnames[0]

            image = image.squeeze(0)
            image = image.unsqueeze(1)
            output = model['cnn'](image)

            task = ['brain', 'knee']
            task = task[torch.argmax(output, dim=1).item()]

            if task == 'brain':
                uniform_height = 384
            else:
                uniform_height = 416
            if uniform_height < kspace.shape[-3]:
                image = ifft2c(kspace)
                h_from = (image.shape[-3] - uniform_height) // 2
                h_to = h_from + uniform_height
                image = image[..., h_from:h_to, :, :]
                kspace = fft2c(image)

            num_slices = kspace.shape[1]
            start_adj, end_adj = -(num_adj_slices[task][acc] // 2), num_adj_slices[task][acc] // 2 + 1
            for slice_idx in range(num_slices):
                sliced_kspace = []
                for slice_adj_idx in [min(max(i+slice_idx, 0), num_slices-1) for i in range(start_adj, end_adj)]:
                    sliced_kspace.append(kspace[:, slice_adj_idx])
                sliced_kspace = torch.concatenate(sliced_kspace, dim=1)
                output = model[task][acc](sliced_kspace, mask)
                reconstructions[fname][slice_idx] = output[0].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions


def forward(args):
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    print ('Current cuda device ', torch.cuda.current_device())

    checkpoint = {
        'cnn': torch.load(args.cnn_checkpoint, map_location=device, weights_only=False),
        'brain': {
            'acc4': torch.load(args.brain_acc4_checkpoint, map_location=device, weights_only=False),
            'acc8': torch.load(args.brain_acc8_checkpoint, map_location=device, weights_only=False),
        },
        'knee': {
            'acc4': torch.load(args.knee_acc4_checkpoint, map_location=device, weights_only=False),
            'acc8': torch.load(args.knee_acc8_checkpoint, map_location=device, weights_only=False),
        },
    }

    model = defaultdict(dict)
    num_adj_slices = defaultdict(dict)
    model['cnn'] = resolve_class(checkpoint['cnn']['args'].model_name)().to(device=device)

    for task in ['brain', 'knee']:
        for acc in ['acc4', 'acc8']:
            saved_args = checkpoint[task][acc]['args']
            num_adj_slices[task][acc] = saved_args.num_adj_slices if hasattr(saved_args, 'num_adj_slices') else 1
            model_name = saved_args.model_name
            ModelClass = resolve_class(model_name)

            model_instance = ModelClass(
                num_cascades=saved_args.num_cascades,
                num_adj_slices=saved_args.num_adj_slices,
                n_feat0=saved_args.n_feat0,
                feature_dim=saved_args.feature_dim,
                prompt_dim=saved_args.prompt_dim,
                sens_n_feat0=saved_args.sens_n_feat0,
                sens_feature_dim=saved_args.sens_feature_dim,
                sens_prompt_dim=saved_args.sens_prompt_dim,
                len_prompt=saved_args.len_prompt,
                prompt_size=saved_args.prompt_size,
                n_enc_cab=saved_args.n_enc_cab,
                n_dec_cab=saved_args.n_dec_cab,
                n_skip_cab=saved_args.n_skip_cab,
                n_bottleneck_cab=saved_args.n_bottleneck_cab,
                n_buffer=saved_args.n_buffer,
                n_history=saved_args.n_history,
                no_use_ca=saved_args.no_use_ca,
                learnable_prompt=saved_args.learnable_prompt,
                adaptive_input=saved_args.adaptive_input,
                use_sens_adj=saved_args.use_sens_adj,
                compute_sens_per_coil=False,
            ).to(device=device)
            
            model[task][acc]=model_instance

    print(f"cnn: epoch={checkpoint['cnn']['epoch']}")
    model['cnn'].load_state_dict(checkpoint['cnn']['module'])
    for task in ['brain', 'knee']:
        for acc in ['acc4', 'acc8']:
            print(f"{task}_{acc}: epoch={checkpoint[task][acc]['epoch']}")
            model[task][acc].load_state_dict(checkpoint[task][acc]['module'])
    
    forward_loader = create_data_loaders(data_path=args.data_path, args=Namespace(seed=430), data_type='test', slicedata='TestSliceData')
    reconstructions = test(model, forward_loader, num_adj_slices)
    save_reconstructions(reconstructions, args.forward_dir, inputs=None)