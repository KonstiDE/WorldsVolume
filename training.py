import os
import tempfile
import random

import numpy as np
import ray
import torch
import shutup
import sys

import torch.optim as optim
import statistics as s

from ray.tune import CheckpointConfig, RunConfig
from ray.tune.logger import TBXLoggerCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

import config.config as cfg
from model.unet.unet_model import UNET

from provider.dataset_provider import get_dataset, get_loader_from_dataset
from model.dense121_unet.dense121_unet_model import DenseNet121

from metrics.mean_errors import mae, r_mse, medae
from tqdm import tqdm as loop_tqdm

from ray import tune
from ray import train as raytrain
from ray.train import Checkpoint
from ray.experimental.tqdm_ray import tqdm

from utils.stopper import PatienceStopper

sys.path.append(os.getcwd())
device = cfg.get_device()
shutup.please()


def train(loader, loss_fn, optimizer, scaler, model):
    torch.enable_grad()
    model.train()

    running_loss = []
    running_mae = []
    running_medae = []
    running_mse = []
    running_rmse = []

    loop = loop_tqdm(loader)

    for (data, target, _) in loop:
        optimizer.zero_grad(set_to_none=True)
        data = data.to(device)

        data = model(data)

        target = target.to(device)

        with torch.amp.autocast("cuda"):
            loss = loss_fn(data, target)
            loss_value = loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        mse, rmse = r_mse(data, target)

        running_mae.append(mae(data, target).item())
        running_medae.append(medae(data, target).item())
        running_mse.append(mse.item())
        running_rmse.append(rmse.item())

        running_loss.append(loss_value)

        loop.set_postfix_str(f"Loss: {loss.item():.4f}")

    return s.mean(running_loss), s.mean(running_mae), s.mean(running_medae), s.mean(running_mse), s.mean(running_rmse)


def valid(loader, loss_fn, model):
    model.eval()

    running_loss = []
    running_mae = []
    running_medae = []
    running_mse = []
    running_rmse = []

    loop = loop_tqdm(loader)

    for (data, target, _) in loop:
        data = data.to(device)

        data = model(data)

        target = target.to(device)

        with torch.no_grad():
            loss = loss_fn(data, target)
            loss_value = loss.item()

        mse, rmse = r_mse(data, target)

        running_mae.append(mae(data, target).item())
        running_medae.append(medae(data, target).item())
        running_mse.append(mse.item())
        running_rmse.append(rmse.item())

        running_loss.append(loss_value)

        loop.set_postfix_str(f"Loss: {loss.item():.4f}")

    return s.mean(running_loss), s.mean(running_mae), s.mean(running_medae), s.mean(running_mse), s.mean(running_rmse)


def run(ray_config, data_cache):
    torch.cuda.empty_cache()

    set_reproducibility()

    model = UNET(in_channels=20, out_channels=1)

    optimizer = optim.AdamW(model.parameters(), lr=ray_config["lr"])
    loss_fn = torch.nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda")

    epochs_done = 0

    model.to(device)

    train_tensors = ray.get(data_cache["train"])
    val_tensors = ray.get(data_cache["val"])

    train_loader = get_loader_from_dataset(train_tensors, ray_config["batch_size"], cfg.num_workers(), cfg.pin_memory())
    val_loader = get_loader_from_dataset(val_tensors, ray_config["batch_size"], cfg.num_workers(), cfg.pin_memory())

    for epoch in range(epochs_done + 1, 101):
        t_loss, t_mae, t_medae, t_mse, t_rmse = train(
            train_loader, loss_fn, optimizer, scaler, model)

        v_loss, v_mae, v_medae, v_mse, v_rmse = valid(
            val_loader, loss_fn, model)

        # Ray report
        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        metrics = {
            "training_loss": t_loss,
            "validation_loss": v_loss,
            "training_mae": t_mae,
            "training_medae": t_medae,
            "training_mse": t_mse,
            "training_rmse": t_rmse,
            "validation_mae": v_mae,
            "validation_medae": v_medae,
            "validation_mse": v_mse,
            "validation_rmse": v_rmse
        }

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                checkpoint_data,
                os.path.join(temp_checkpoint_dir, "model_epoch{}.pt".format(epoch)),
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            ray.tune.report(metrics, checkpoint=checkpoint)


def set_reproducibility():
    torch.manual_seed(251199)
    np.random.seed(251199)
    random.seed(251199)
    torch.use_deterministic_algorithms(True)



if __name__ == '__main__':
    print(f"Loading data into shared memory...")

    ray.init(config={
        "object_spilling_config": '{"type":"smart_open","params":{"uri":"A:/tmp"}}'
    }, _temp_dir="A:/tmp")

    shared_data_cache = {}

    train_ds = get_dataset(cfg.get_train_path(), True, load_amount=0)
    shared_data_cache["train"] = ray.put(train_ds)
    del train_ds

    valid_ds = get_dataset(cfg.get_validation_path(), False, load_amount=0)
    shared_data_cache["val"] = ray.put(valid_ds)
    del valid_ds

    optuna_search = OptunaSearch(metric="validation_loss", mode="min")

    universal_lr_space = tune.loguniform(1e-6, 1e-1)
    asha_scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="validation_loss",
        mode="min",
        max_t=100,
        grace_period=2,
        reduction_factor=3
    )

    trainable = tune.with_resources(
        tune.with_parameters(run, data_cache=shared_data_cache),
        resources={"cpu": 16, "gpu": 2}
    )

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            num_samples=30,
            scheduler=asha_scheduler,
            search_alg=optuna_search
        ),
        run_config=RunConfig(
            name="ray_results",
            callbacks=[TBXLoggerCallback()],
            stop=PatienceStopper(
                metric="validation_loss",
                mode="min",
                patience=5
            ),
            storage_path="A:\\volume",
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="validation_loss",
                checkpoint_score_order="min"
            )
        ),
        param_space={
            "batch_size": 4,
            "lr": universal_lr_space
        },
    )

    results = tuner.fit()

    best_result = results.get_best_result("validation_loss", "min", "all")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["validation_loss"]))
