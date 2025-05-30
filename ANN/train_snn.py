import os
import random
import argparse
import yaml
import numpy as np
import scipy.stats as st
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, recall_score, f1_score, mean_absolute_error
from models import ConvConv, SpikingConvConv, collect_max_act

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
    torch.set_float32_matmul_precision('high')
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    # Reproducibility
    SEED = cfg.get('seed', 0)
    seed_everything(SEED)

    # Hyperparameters
    EPOCHS = cfg['training']['epochs']
    BATCH_SIZE = cfg['training']['batch_size']
    CNN_FILTERS = cfg['model']['cnn_filters']

    ATT_HOPS = cfg['model']['attention_num_hops']
    ATT_SIZE = cfg['model']['attention_size']

    LEARNING_RATE = cfg['training']['learning_rate']
    PATIENCE = cfg['training']['early_stopping_patience']
    MODE = cfg['data']['mode']


    BASE_DIR = cfg['output']['base_dir']
    SAVE_DIR = cfg['output']['save_dir']
    os.makedirs(SAVE_DIR, exist_ok=True)


    DATA_FILES = cfg['data']['files']

    for data_file in DATA_FILES:
        data=np.load(os.path.join(BASE_DIR, data_file), allow_pickle=True)
        X=np.squeeze(data['X'],1); y_one=data['y']; folds=data['folds']
        num_labels=y_one.shape[1]; y=np.argmax(y_one,1)

        accs, recs, f1s = [], [], []
        for i,(tr,te) in enumerate(folds):

            # wandb_logger = WandbLogger(
            #     project=cfg['wandb']['project'],
            #     name=f'CC_{MODE}_run_fold_{i}',
            #     save_dir=SAVE_DIR,
            #     log_model=True
            # )

            Xt, yt = X[tr], y[tr]; Xv, yv = X[te], y[te]
            def trim(a): return a[:len(a)//BATCH_SIZE*BATCH_SIZE]
            Xt, yt=trim(Xt), trim(yt); Xv, yv=trim(Xv), trim(yv)
            Xt=np.expand_dims(Xt,3); Xv=np.expand_dims(Xv,3)

            tr_loader=prepare_loader(Xt, yt, BATCH_SIZE, shuffle=True)
            val_loader=prepare_loader(Xv, yv, BATCH_SIZE)

            model=ConvConv(
                input_shape=Xt.shape[1:], num_labels=num_labels, size=ATT_SIZE,
                num_conv_filters=CNN_FILTERS, num_hops=ATT_HOPS,
                learning_rate=LEARNING_RATE
            )
            checkpoint_cb = ModelCheckpoint(dirpath=SAVE_DIR, filename=f'{MODE}_fold_{i}_CC',
                                            save_top_k=1, monitor='val_acc', mode='max')
            
            es_cb = EarlyStopping(monitor='val_acc', patience=PATIENCE, mode='max')

            trainer = pl.Trainer(
                max_epochs=EPOCHS,
                accelerator=cfg['training']['accelerator'],
                devices=cfg['training']['devices'],
                callbacks=[checkpoint_cb, es_cb],
                default_root_dir=SAVE_DIR,
                deterministic=True
            )

            trainer.fit(model, tr_loader, val_loader)
            
            # ---------- (A) load best ANN checkpoint ----------
            best_ann = ConvConv.load_from_checkpoint(checkpoint_cb.best_model_path).cuda()
            best_ann.eval()

            # ---------- (B) calibrate on a small slice of training data ----------
            cal_loader = prepare_loader(Xt[:10*BATCH_SIZE], yt[:10*BATCH_SIZE],   # 10 mini-batches
                                        batch_size=BATCH_SIZE)
            max_act = collect_max_act(best_ann, cal_loader)                       # ≲ 1 s/GPU

            # we need a dummy entry for the network input range:
            max_act['input'] = torch.tensor([1.0])                                # inputs are in [0,1]

            # ---------- (C) convert & evaluate ----------
            spike_net = SpikingConvConv(best_ann, max_act, T=16, Q=1.3).cuda()
            spike_net.eval()

            preds_spk, targs_spk = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    out_spk = spike_net(xb.cuda())
                    preds_spk.append(torch.argmax(out_spk, 1).cpu().numpy())
                    targs_spk.append(yb.numpy())
            preds_spk = np.concatenate(preds_spk); targs_spk = np.concatenate(targs_spk)
            acc_spk  = accuracy_score(targs_spk, preds_spk)
            rec_spk  = recall_score(targs_spk, preds_spk, average='macro')
            f1_spk   = f1_score(targs_spk, preds_spk, average='macro')
            print(f"[SNN] Fold {i}  Acc:{acc_spk:.4f}  Recall:{rec_spk:.4f}  F1:{f1_spk:.4f}")
           

        def ci90(arr): return st.t.interval(0.9, len(arr)-1, loc=np.mean(arr), scale=st.sem(arr))
        print(f"Mean Acc:{np.mean(accs):.4f}, CI:{ci90(accs)}")
        print(f"Mean Recall:{np.mean(recs):.4f}, CI:{ci90(recs)}")
        print(f"Mean F1:{np.mean(f1s):.4f}, CI:{ci90(f1s)}")

    