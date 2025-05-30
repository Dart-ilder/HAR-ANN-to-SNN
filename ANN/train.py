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

    # Hyperparameters
    EPOCHS = cfg['training']['epochs']
    BATCH_SIZE = cfg['training']['batch_size']
    CNN_FILTERS = cfg['model']['cnn_filters']
    # Self-attention params from layers.py
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
                # logger=wandb_logger,
                default_root_dir=SAVE_DIR,
                deterministic=True
            )

            trainer.fit(model, tr_loader, val_loader)
            

            best=ConvConv.load_from_checkpoint(checkpoint_cb.best_model_path).cuda()
            best.eval()
            preds, targs = [], []
            for xbatch,ybatch in val_loader:
                out=best(xbatch.cuda())
                preds.append(torch.argmax(out,1).cpu().numpy())
                targs.append(ybatch.numpy())
            preds=np.concatenate(preds); targs=np.concatenate(targs)

            # wandb_logger.experiment.finish()

            a=accuracy_score(targs,preds); r=recall_score(targs,preds,average='macro'); f1s_co=f1_score(targs,preds,average='macro'); mae=mean_absolute_error(targs,preds)
            accs.append(a); recs.append(r); f1s.append(f1s_co)
            print(f"Fold {i} -- Acc:{a:.4f}, Recall:{r:.4f}, F1:{f1s_co:.4f}, MAE:{mae:.4f}")

           

        def ci90(arr): return st.t.interval(0.9, len(arr)-1, loc=np.mean(arr), scale=st.sem(arr))
        print(f"Mean Acc:{np.mean(accs):.4f}, CI:{ci90(accs)}")
        print(f"Mean Recall:{np.mean(recs):.4f}, CI:{ci90(recs)}")
        print(f"Mean F1:{np.mean(f1s):.4f}, CI:{ci90(f1s)}")

    