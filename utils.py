import importlib
import faiss
from recbole.data.utils import create_dataset as create_recbole_dataset


def parse_faiss_index(pq_index):
    vt = faiss.downcast_VectorTransform(pq_index.chain.at(0))
    assert isinstance(vt, faiss.LinearTransform)
    opq_transform = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)

    ivf_index = faiss.downcast_index(pq_index.index)
    invlists = faiss.extract_index_ivf(ivf_index).invlists
    ls = invlists.list_size(0)
    pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
    pq_codes = pq_codes.reshape(-1, invlists.code_size)

    centroid_embeds = faiss.vector_to_array(ivf_index.pq.centroids)
    centroid_embeds = centroid_embeds.reshape(ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub)

    coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
    coarse_embeds = faiss.rev_swig_ptr(coarse_quantizer.get_xb(), ivf_index.pq.M * ivf_index.pq.dsub)
    coarse_embeds = coarse_embeds.reshape(-1)

    return pq_codes, centroid_embeds, coarse_embeds, opq_transform


def create_dataset(config):
    dataset_module = importlib.import_module('data.dataset')
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        return getattr(dataset_module, config['model'] + 'Dataset')(config)
    else:
        return create_recbole_dataset(config)
