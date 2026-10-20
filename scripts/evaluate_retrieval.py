"""Evaluate aligned query/gallery embeddings using the official item IDs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from fashion_assistant.config import Config

def main():
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=cfg.dataset_root)
    p.add_argument('--embedding-dir', type=Path, default=cfg.output_dir)
    p.add_argument('--top-k', type=int, default=5)
    args = p.parse_args()
    f = args.data_dir / 'Eval' / 'list_eval_partition.txt'
    if not f.exists(): f = args.data_dir / 'eval' / 'list_eval_partition.txt'
    df = pd.read_csv(f, sep=r'\s+', skiprows=1)
    item_map = dict(zip(df.image_name.astype(str), df.item_id.astype(str)))
    q = np.load(args.embedding_dir/'query_embeddings.npy')
    g = np.load(args.embedding_dir/'gallery_embeddings.npy')
    qp = np.load(args.embedding_dir/'query_paths.npy').astype(str)
    gp = np.load(args.embedding_dir/'gallery_paths.npy').astype(str)
    assert len(q) == len(qp) and len(g) == len(gp), 'Embedding/path alignment mismatch'
    official_q = set(df.loc[df.evaluation_status=='query','image_name'].astype(str))
    official_g = set(df.loc[df.evaluation_status=='gallery','image_name'].astype(str))
    assert len(set(qp))==len(qp) and len(set(gp))==len(gp), 'Duplicate paths'
    assert set(qp)==official_q and set(gp)==official_g, 'Expected full official query/gallery partitions'
    assert np.isfinite(q).all() and np.isfinite(g).all()
    assert np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-3)
    assert np.allclose(np.linalg.norm(g,axis=1),1,atol=1e-3)
    gallery_ids = np.array([item_map[x] for x in gp])
    counts = pd.Series(gallery_ids).value_counts()
    k = args.top_k
    precision, recall, hit1, hitk = [], [], [], []
    for start in range(0,len(q),128):
        sims = q[start:start+128] @ g.T
        # Sorted top-k; full sort is intentionally kept for readability
        top = np.argsort(sims,axis=1)[:,-k:][:,::-1]
        for offset, indices in enumerate(top):
            item = item_map[qp[start+offset]]
            if item not in counts: continue
            correct = gallery_ids[indices] == item
            precision.append(correct.sum()/k)
            recall.append(correct.sum()/counts[item])
            hit1.append(int(correct[0]))
            hitk.append(int(correct.any()))
    print(f'Evaluated queries: {len(precision)}')
    for name, vals in [('Precision@K',precision),('Recall@K',recall),('Hit@1',hit1),('Hit@K',hitk)]:
        print(f'{name} (K={k}): {np.mean(vals):.4f}')
if __name__=='__main__': main()
