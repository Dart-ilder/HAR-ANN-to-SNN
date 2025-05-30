import os
import random
import argparse
import yaml
import numpy as np
import scipy.stats as st
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, f1_score, mean_absolute_error
from models import ConvAttention, ConvConv
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    
def prepare_loader(X, y, batch_size, shuffle=False):
    # convert to float32/long tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=24)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    # Reproducibility
    SEED = cfg.get('seed', 0)
    seed_everything(SEED)

    BATCH_SIZE = cfg['training']['batch_size']
    MODE = cfg['data']['mode']
    BASE_DIR = cfg['output']['base_dir']
    DATA_FILES = cfg['data']['files']
    TYPE = 'CC'

    for data_file in DATA_FILES:
        data=np.load(os.path.join(BASE_DIR, data_file), allow_pickle=True)
        X=np.squeeze(data['X'],1); y_one=data['y']; folds=data['folds']
        num_labels=y_one.shape[1]; y=np.argmax(y_one,1)

        accs, recs, f1s = [], [], []
        for i,(tr,te) in enumerate(folds):

            Xt, yt = X[tr], y[tr]; Xv, yv = X[te], y[te]
            def trim(a): return a[:len(a)//BATCH_SIZE*BATCH_SIZE]
            Xt, yt=trim(Xt), trim(yt); Xv, yv=trim(Xv), trim(yv)
            Xt=np.expand_dims(Xt,3); Xv=np.expand_dims(Xv,3)

            val_loader=prepare_loader(Xv, yv, BATCH_SIZE)      

            best=ConvConv.load_from_checkpoint(f"./chkpts/{MODE}_fold_{i}_{TYPE}").cuda()
            best.eval()
            preds, targs = [], []
            for xbatch,ybatch in val_loader:
                out=best(xbatch.cuda())
                preds.append(torch.argmax(out,1).cpu().numpy())
                targs.append(ybatch.numpy())
            preds=np.concatenate(preds); targs=np.concatenate(targs)

            a=accuracy_score(targs,preds); r=recall_score(targs,preds,average='macro'); f1s_co=f1_score(targs,preds,average='macro'); mae=mean_absolute_error(targs,preds)
            accs.append(a); recs.append(r); f1s.append(f1s_co)
            print(f"Fold {i} -- Acc:{a:.4f}, Recall:{r:.4f}, F1:{f1s_co:.4f}, MAE:{mae:.4f}") 

        def ci90(arr): return st.t.interval(0.9, len(arr)-1, loc=np.mean(arr), scale=st.sem(arr))
        print(f"Mean Acc:{np.mean(accs):.4f}, CI:{ci90(accs)}")
        print(f"Mean Recall:{np.mean(recs):.4f}, CI:{ci90(recs)}")
        print(f"Mean F1:{np.mean(f1s):.4f}, CI:{ci90(f1s)}")
