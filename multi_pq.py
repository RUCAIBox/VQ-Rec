import os
import argparse
import numpy as np
import faiss
from tqdm import tqdm


def parse_args():
    # Basic
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Food,Home,CDs,Kindle,Movies')
    parser.add_argument('--input_path', type=str, default='dataset/pretrain/')
    parser.add_argument('--output_path', type=str, default='dataset/pretrain/')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--suffix', type=str, default='feat1CLS')
    parser.add_argument('--plm_size', type=int, default=768)

    # PQ
    parser.add_argument("--subvector_num", type=int, default=32, help='16/24/32/48/64/96')
    parser.add_argument("--n_centroid", type=int, default=8)
    parser.add_argument("--use_gpu", type=int, default=True)
    parser.add_argument("--strict", type=int, default=True)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    dataset_names = args.dataset.split(',')
    print('Convert dataset: ')
    print(' Dataset: ', dataset_names)
    
    short_name = ''.join([_[0] for _ in dataset_names])
    print(' Short name: ', short_name)

    if args.strict:
        item_set = [set() for i in range(len(dataset_names))]
        inter_path = os.path.join(args.input_path, short_name, f'{short_name}.train.inter')
        print(f'Strict Mode: Loading training data from [{inter_path}]')
        with open(inter_path, 'r', encoding='utf-8') as file:
            file.readline()
            for line in tqdm(file):
                user_id, item_seq, item_id = line.strip().split('\t')
                did, pure_item_id = item_id.split('-')
                item_seq = [_.split('-')[-1] for _ in item_seq.split(' ')]
                for idx in item_seq + [pure_item_id]:
                    item_set[int(did)].add(int(idx))
        filter_id_list = []
        with open(os.path.join(args.input_path, short_name, f'{short_name}.filtered_id'), 'w', encoding='utf-8') as file:
            for did in range(len(dataset_names)):
                print(f'Strict Mode: Writing [{dataset_names[did]}] indexes down')
                filter_id = np.array(sorted(list(item_set[did])))
                filter_id_list.append(filter_id)
                for iid in filter_id.tolist():
                    file.write(f'{did}-{iid}\n')
    
    feat_list = []
    for did, ds in enumerate(dataset_names):
        feat_path = os.path.join(args.input_path, short_name, f'{ds}.{args.suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, args.plm_size)
        print(f'Load {loaded_feat.shape} from {feat_path}.')
        if args.strict:
            feat_list.append(loaded_feat[filter_id_list[did]])
        else:
            feat_list.append(loaded_feat)

    merged_feat = np.concatenate(feat_list, axis=0)
    print('Merged feature: ', merged_feat.shape)

    save_index_path = os.path.join(
        args.output_path,
        short_name,
        f"{short_name}.OPQ{args.subvector_num},IVF1,PQ{args.subvector_num}x{args.n_centroid}{'.strict' if args.strict else ''}.index")

    if args.use_gpu:
        res = faiss.StandardGpuResources()
        res.setTempMemory(1024 * 1024 * 512)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = args.subvector_num >= 56
    faiss.omp_set_num_threads(32)

    index = faiss.index_factory(args.plm_size,
        f"OPQ{args.subvector_num},IVF1,PQ{args.subvector_num}x{args.n_centroid}", faiss.METRIC_INNER_PRODUCT)
    index.verbose = True
    if args.use_gpu:
        index = faiss.index_cpu_to_gpu(res, args.gpu_id, index, co)
    index.train(merged_feat)
    index.add(merged_feat)
    if args.use_gpu:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, save_index_path)
