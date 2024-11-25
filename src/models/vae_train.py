from utils import get_logger
import torch
import numpy as np
from .vae import CVAE, vae_loss
from .vae_utils import RNADataset, get_dataloader, Metrics
from tqdm.auto import tqdm
from sklearn.utils.class_weight import compute_class_weight

logger = get_logger(__name__)

def train(X_train, y_train, X_val, y_val, config, model_path, pbar=False):
    epochs = config['num_epochs']
    lr = config['learning_rate']
    weight_decay = 0.001
    
    train_dataset = RNADataset(X_train, np.array(y_train))
    val_dataset = RNADataset(X_val, np.array(y_val))

    ## init loaders ###
    # use weighted sampling to oversample minority classes
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    sample_weights = np.array([class_weights_dict[cls] for cls in y_train])
    train_dataloader = get_dataloader(train_dataset, config, weights=sample_weights)
    val_dataloader = get_dataloader(val_dataset, config, training=False)


    ### init model ###
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Training on {device}")
    model = CVAE(
        data_dim=2500, 
        latent_dim=512,
        dropout=0.25,
        num_classes=len(np.unique(y_train)),
        class_emb_dim=64
    ).to(device)
    print(model)

    ## optim ##
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.001, total_steps=epochs*2)
    beta = np.linspace(0.1, 1, 10)


    train_metrics = Metrics()
    eval_metrics = Metrics(desc="Eval")

    for epoch in range(epochs):
        ### Train ###
        model.train()

        for i, (x, y) in tqdm(enumerate(train_dataloader, 1), desc="Training", disable=not pbar):
            opt.zero_grad()
            x, y = x.to(device), y.to(device)
            # recon_mu, recon_logvar, q = model(x, y)
            reconstructed, mu, logvar = model(x, y)

            if epoch >= len(beta):
                b = beta[-1]
            else:
                b = beta[epoch]
            # loss, recon_loss, kl_loss = vae_loss(recon_mu, recon_logvar, x, q, b)
            loss, recon_loss, kl_loss = vae_loss(reconstructed, x, mu, logvar, beta=b)

            loss.backward()
            train_metrics.update(kl_loss, recon_loss, loss.item())
            
            opt.step()

        scheduler.step()
        train_metrics.end_epoch()

        ### EVAL ###
        model.eval()
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(val_dataloader, 1), desc="Eval", disable=not pbar):
                x, y = x.to(device), y.to(device)
                # recon_mu, recon_logvar, q = model(x, y)
                reconstructed, mu, logvar = model(x, y)
                # loss, recon_loss, kl_loss = vae_loss(recon_mu, recon_logvar, x, q, b)
                loss, recon_loss, kl_loss = vae_loss(reconstructed, x, mu, logvar, beta=b)
                eval_metrics.update(kl_loss, recon_loss, loss.item())
        eval_metrics.end_epoch()
        if model_path and eval_metrics.epochs_since_best == 0:
            torch.save(model.state_dict(), model_path)
        if eval_metrics.early_stopping():
            logger.info("Stopping")
            break
        print()
    
    return model
