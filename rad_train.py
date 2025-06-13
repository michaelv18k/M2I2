# import argparse
# import os
# import ruamel.yaml as yaml
# import time
# import datetime
# import json
# from pathlib import Path
# import torch
# from models.model_vqa import M2I2
# from models.vit import interpolate_pos_embed
# from models.tokenization_bert import BertTokenizer
# import utils
# from dataset.utils import save_result
# from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn
# from scheduler import create_scheduler
# from optim import create_optimizer


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    for i, (image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device, non_blocking=True), weights.to(device, non_blocking=True)

        question_input = tokenizer(question, padding='longest', truncation=True,
                                   max_length=25, return_tensors="pt").to(device)
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(image, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
# def evaluation(model, data_loader, tokenizer, device, config):
#     config['k_test'] = min(config.get('k_test', 5), 1)  # reduce top-k to 1 to save memory
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Generate VQA test result:'
#     print_freq = 50
#     result = []

#     answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]
#     answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)

#     for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         image = image.to(device, non_blocking=True)
#         question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)

#         torch.cuda.empty_cache()
#         topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])
#         torch.cuda.empty_cache()

#         for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
#             ques_id = int(ques_id.item())
#             _, pred = topk_prob.max(dim=0)
#             result.append({"qid": ques_id, "answer": data_loader.dataset.answer_list[topk_id[pred]]})

#     return result

import argparse
import os
import ruamel.yaml as yaml
import time
import datetime
import json
from pathlib import Path
import torch
from models.model_vqa import M2I2
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn
from scheduler import create_scheduler
from optim import create_optimizer
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import numpy as np


def visualize_prediction_with_attention(image_tensor, question, answer, attention_map, save_path):
    image = image_tensor.cpu().squeeze(0)
    image = T.ToPILImage()(image)
    image_np = np.array(image).astype(np.float32) / 255.0

    attention_map = attention_map.squeeze().cpu().numpy()
    attention_map = np.mean(attention_map, axis=0)
    attention_map = np.maximum(attention_map, 0)
    attention_map /= attention_map.max() + 1e-8
    attention_map_resized = np.uint8(255 * attention_map)
    attention_map_resized = Image.fromarray(attention_map_resized).resize(image.size, resample=Image.BILINEAR)
    attention_map_resized = np.array(attention_map_resized)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_np, alpha=0.6)
    plt.imshow(attention_map_resized, cmap='jet', alpha=0.4)
    plt.title(f"Q: {question}\nA: {answer}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def enable_attention_saving(model):
    if hasattr(model, 'text_encoder') and hasattr(model.text_encoder, 'encoder'):
        for layer in model.text_encoder.encoder.layer:
            if hasattr(layer, 'crossattention'):
                layer.crossattention.self.save_attention = True


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    config['k_test'] = min(config.get('k_test', 5), 1)
    model.eval()
    enable_attention_saving(model)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    result = []

    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)

    vis_dir = os.path.join(args.output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)


    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)

        torch.cuda.empty_cache()
        topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])
        torch.cuda.empty_cache()

        for idx, (ques_id, topk_id, topk_prob) in enumerate(zip(question_id, topk_ids, topk_probs)):
            ques_id = int(ques_id.item())
            _, pred = topk_prob.max(dim=0)
            answer_text = data_loader.dataset.answer_list[topk_id[pred]]
            result.append({"qid": ques_id, "answer": answer_text})

            if idx == 0:
                attn_map = model.text_encoder.encoder.layer[-1].crossattention.self.get_attention_map()
                if attn_map is not None:
                    save_path = os.path.join(vis_dir, f"qid_{ques_id}.png")
                    visualize_prediction_with_attention(image[0], question[0], answer_text, attn_map, save_path)

    return result



def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    utils.set_seed(seed)

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    print("Creating vqa rad datasets")
    datasets = create_dataset('rad', config)
    print('train dataset size: ', len(datasets[0]))
    print('test dataset size: ', len(datasets[1]))

    samplers = create_sampler(datasets, [True, False], utils.get_world_size(), utils.get_rank()) if args.distributed else [None, None]

    train_loader, test_loader = create_loader(datasets, samplers,
                                              batch_size=[config['batch_size_train'], config['batch_size_test']],
                                              num_workers=[4, 4], is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn, None])

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    print("Creating model")
    model = M2I2(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(device)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

            for key in list(state_dict.keys()):
                if 'bert' in key:
                    encoder_key = key.replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                if 'text_encoder' in key:
                    if 'layer' in key:
                        encoder_keys = key.split('.')
                        layer_num = int(encoder_keys[4])
                        if layer_num < 6:
                            del state_dict[key]
                            continue
                        else:
                            decoder_layer_num = (layer_num - 6)
                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = '.'.join(encoder_keys)
                    else:
                        encoder_key = key
                    decoder_key = encoder_key.replace('text_encoder', 'text_decoder')
                    state_dict[decoder_key] = state_dict[key]
                    del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("\nStart testing\n" if args.evaluate else "\nStart training\n")
    start_time = time.time()

    loss_list = []

    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)

            torch.cuda.empty_cache()
            vqa_result = evaluation(model, test_loader, tokenizer, device, config)
            torch.cuda.empty_cache()

        if args.evaluate:
            break

        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
            loss_list.append(log_stats)
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            prefix = args.checkpoint.split('/')[-1].split('.')[0]
            if epoch >= 20:
                torch.save(save_obj, os.path.join(args.output_dir, f'{prefix}_rad_{epoch:02d}.pth'))

            save_result(vqa_result, args.result_dir, f'{prefix}_vqa_result_{epoch}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/RAD.yaml')
    parser.add_argument('--checkpoint', default='./pretrain/2022-09-11/med_pretrain_29.pth')
    parser.add_argument('--output_dir', default='output/rad')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    print("config: ", config)
    print("args: ", args)
    main(args, config)
