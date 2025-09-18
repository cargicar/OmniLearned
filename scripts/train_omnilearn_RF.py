
import rootutils
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from rectified_flow.rectified_flow import RectifiedFlow

rootutils.setup_root(__file__, pythonpath=True)


#from dataloader import load_data
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
#from pytorch_optimizer import Lion
#from lion_pytorch import Lion
from diffusers.optimization import get_cosine_schedule_with_warmup

from src.models.omnilearnedv2 import PET3
from src.models.omnilearned import PET2
from src.diffusion.edm import EDM
from src.data.dataset import HDF5Dataset, pad_collate_fn, PklDataset, ShapeNetCore 

from scripts.utils import (
    is_master_node,
    ddp_setup,
    get_param_groups,
    CLIPLoss,
    get_checkpoint_name,
)
import time
import os
import torch.amp as amp

torch.set_float32_matmul_precision("high")
torch._dynamo.config.verbose = False

def parse_arguments():
    """
    Parses command-line arguments for the model training script.

    Returns:
        argparse.Namespace: An object containing all the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run model training with specified configurations.")

    # --- General/Output Arguments ---
    
    parser.add_argument("--path", type=str, default='/pscratch/sd/c/ccardona/datasets/G4_individual_sims_pkl_test',
                         help="Base path to the dataset directory.")
    
    #parser.add_argument("--path", type=str, default="/pscratch/sd/c/ccardona/datasets/G4_h5/all_sims_combined.h5",
    #                     help="Base path to the dataset directory.")
    parser.add_argument("--outdir", type=str, default="/pscratch/sd/c/ccardona/models/G4/",
                          help="Output directory for logs, checkpoints, and results.")
    #parser.add_argument("--outdir", type=str, default="/home/carlos/Rnet_local/saved_models",
    #                    help="Output directory for logs, checkpoints, and results.")
    parser.add_argument("--save_tag", type=str, default="detector_cats_RF",
                        help="Tag to append to saved files (e.g., model checkpoints, logs).")
    parser.add_argument("--pretrain_tag", type=str, default="pretrain",
                        help="Tag to use when loading pre-trained models.")
    parser.add_argument("--dataset", type=str, default="G4",
                        help="Name of the dataset to use (e.g., 'G4').")
    parser.add_argument("--wandb", action="store_true", # Use store_true for boolean flags
                        help="Enable Weights & Biases logging.")
   
    # --- Training State Arguments ---
    parser.add_argument("--fine_tune", action="store_true",
                        help="Enable fine-tuning mode (loads pre-trained weights and adjusts learning rate).")
    parser.add_argument("--resuming", action="store_true",
                        help="Resume training from the latest checkpoint in outdir/save_tag.")

    # --- Data/Feature Arguments ---

    parser.add_argument("--num_feat", type=int, default=4,
                        help="Number of features per particle/vector (e.g., 4 for 4-vectors).")
    parser.add_argument("--conditional", action="store_true",
                        help="Enable conditional generation/training.")
    
    parser.add_argument("--use_interaction", action="store_true",
                        help="Enable interaction block.")
    parser.add_argument("--num_cond", type=int, default=3,
                        help="Number of conditioning features/dimensions.")
    parser.add_argument("--use_pid", action="store_true",
                        help="Use Particle ID (PID) as an input feature.")
    parser.add_argument("--pid_idx", type=int, default=-1,
                        help="Index of the PID feature in the input data (if use_pid is True).")
    parser.add_argument("--use_add", action="store_true",
                        help="Use additional features.")
    parser.add_argument("--num_add", type=int, default=4,
                        help="Number of additional features.")
    parser.add_argument("--use_clip", action="store_true",
                        help="Enable gradient clipping.")
    parser.add_argument("--use_event_loss", action="store_true",
                        help="Enable event-level loss calculation.")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of output classes for classification tasks.")
    parser.add_argument("--num_gap_classes", type=int, default=4,
                        help="Number of classes for detector classification tasks.")
    parser.add_argument("--mode", type=str, default="generator",
                        choices=["classifier", "generator", "other_mode_if_any"], # Add valid choices
                        help="Operating mode of the model (e.g., 'classifier', 'generator').")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of worker processes for data loading.")


    # --- Training Hyperparameters ---
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size for training.")
    parser.add_argument("--iterations", type=int, default=-1,
                        help="Number of training iterations. If -1, run for specified epochs.")
    parser.add_argument("--epoch", type=int, default=1000,
                        help="Number of training epochs.")
    parser.add_argument("--warmup_epoch", type=int, default=1,
                        help="Number of warmup epochs for learning rate scheduling.")
    parser.add_argument("--use_amp", action="store_true",
                        help="Enable Automatic Mixed Precision (AMP) training.")
    parser.add_argument("--optim", type=str, default="adamw",
                        choices=["adamw", "lion", "sgd"], # Example: add common optimizers
                        help="Optimizer to use (e.g., 'lion', 'adamw').")
    parser.add_argument("--b1", type=float, default=0.95,
                        help="Beta1 parameter for Adam-like optimizers.")
    parser.add_argument("--b2", type=float, default=0.98,
                        help="Beta2 parameter for Adam-like optimizers.")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Initial learning rate.")
    parser.add_argument("--lr_factor", type=float, default=10.0,
                        help="Learning rate factor for fine-tuning or scheduling.")
    parser.add_argument("--wd", type=float, default=0.3,
                        help="Weight decay (L2 regularization).")

    # --- Model Architecture Hyperparameters (if applicable, e.g., for a Transformer) ---
    parser.add_argument("--num_transf", type=int, default=6,
                        help="Number of transformer blocks/layers.")
    parser.add_argument("--num_transf_heads", type=int, default=2,
                        help="Number of attention heads in each transformer block.")
    parser.add_argument("--num_tokens", type=int, default=4,
                        help="Number of tokens in the model (e.g., for certain attention mechanisms).")
    parser.add_argument("--num_head", type=int, default=8,
                        help="General number of attention heads (if different from num_transf_heads).")
    parser.add_argument("--K", type=int, default=15,
                        help="K parameter for K-Nearest Neighbors or similar (e.g., for graph construction).")
    parser.add_argument("--base_dim", type=int, default=64,
                        help="Base dimension for model embeddings/features.")
    parser.add_argument("--mlp_ratio", type=int, default=2,
                        help="MLP hidden dimension ratio relative to base_dim.")
    parser.add_argument("--attn_drop", type=float, default=0.1,
                        help="Dropout rate for attention layers.")
    parser.add_argument("--mlp_drop", type=float, default=0.1,
                        help="Dropout rate for MLP layers.")
    parser.add_argument("--feature_drop", type=float, default=0.0,
                        help="Dropout rate for input features.")


    args = parser.parse_args()
    return args


def train_step(
    model,
    dataloader,
    class_cost,
    gen_cost,
    optimizer,
    scheduler,
    epoch,
    device,
    clip_loss=CLIPLoss(),
    use_clip=False,
    use_event_loss=False,
    iterations_per_epoch=-1,
    use_amp=False,
    gscaler=None,
):
    #FIXME hardcoded
    data_shape = (500,4)
    # Initialize RectifiedFlow with custom settings
    rectified_flow = RectifiedFlow(
        data_shape= data_shape,#(32, 32),
        velocity_field=model,
        interp="straight",
        source_distribution="normal",
        # is_independent_coupling=True,
        # train_time_distribution="uniform",
        # train_time_weight="uniform",
        criterion="mse",
        device=device,
    )
    model.train()

    logs_buff = torch.zeros((7), dtype=torch.float32, device=device)
    logs = {}
    logs["loss"] = logs_buff[0].view(-1)
    logs["loss_class"] = logs_buff[1].view(-1)
    logs["loss_gen"] = logs_buff[2].view(-1)
    logs["loss_perturb"] = logs_buff[3].view(-1)
    logs["loss_clip"] = logs_buff[4].view(-1)
    logs["loss_class_event"] = logs_buff[5].view(-1)
    logs["loss_event_perturb"] = logs_buff[6].view(-1)

    if iterations_per_epoch < 0:
        iterations_per_epoch = len(dataloader)
    data_iter = iter(dataloader)

    for batch_idx in range(iterations_per_epoch):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()  # Zero the gradients
        #X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        # y in G4 dataset is hot_encoded, but here it takes int categories.
        #y = torch.argmax(y, dim=1)
        X, energy, y, gap_pid = batch
        X, energy, y, gap_pid = X.to(device), energy.to(device), y.to(device), gap_pid.to(device)
        y = (y == 2).long()
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }
        with amp.autocast(
            "cuda:{}".format(device) if torch.cuda.is_available() else "cpu",
            enabled=use_amp,
        ):
            outputs = model(X, y, gap_pid, energy, **model_kwargs)
            loss = 0
            
            if outputs["y_pred"] is not None:
                if use_event_loss:
                    event_mask = y >= 200
                    if event_mask.any():
                        loss_event = class_cost(
                            outputs["y_pred"][event_mask][:, 200:], y[event_mask] - 200
                        ).mean()
                        logs["loss_class_event"] += loss_event.detach()
                        loss = loss + loss_event
                    if (~event_mask).any():
                        loss_class = class_cost(
                            outputs["y_pred"][~event_mask][:, :200], y[~event_mask]
                        ).mean()
                        logs["loss_class"] += loss_class.detach()
                        loss = loss + loss_class
                else:
                    
                    loss_class = class_cost(outputs["y_pred"], y).mean()
                    #FIXME for G4 use
                    #y_int = torch.argmax(y, dim=1)
                    #loss_class = class_cost(outputs["y_pred"], y_int).mean()
                    loss = loss + loss_class
                    logs["loss_class"] += loss_class.detach()
            if outputs["z_pred"] is not None:
                x_0 = rectified_flow.sample_source_distribution(X.shape[0])
                t = rectified_flow.sample_train_time(X.shape[0])

                loss = rectified_flow.get_loss(
                    x_0=x_0,
                    x_1=X,
                    t=t,
                )

                nonzero = (outputs["v"][:, :, 0] != 0).sum(1)
                # loss_gen = (
                #     gen_cost(outputs["v"], outputs["z_pred"]).sum((1, 2)) / nonzero
                # )
                # loss_gen = loss_gen.mean()
                # loss = loss + loss_gen
                logs["loss_gen"] += loss.detach()
            if outputs["y_perturb"] is not None:
                if use_event_loss:
                    event_mask = y >= 200
                    if event_mask.any():
                        loss_event = torch.mean(
                            outputs["alpha"][event_mask].squeeze(1)
                            * class_cost(
                                outputs["y_perturb"][event_mask][:, 200:],
                                y[event_mask] - 200,
                            )
                        )
                        logs["loss_event_perturb"] += loss_event.detach()
                        loss = loss + loss_event

                    if (~event_mask).any():
                        loss_class = torch.mean(
                            outputs["alpha"][~event_mask].squeeze(1)
                            * class_cost(
                                outputs["y_perturb"][~event_mask][:, :200],
                                y[~event_mask],
                            )
                        )
                        logs["loss_perturb"] += loss_class.detach()
                        loss = loss + loss_class

                else:
                    loss_perturb = torch.mean(
                        outputs["alpha"].squeeze(1)
                        * class_cost(outputs["y_perturb"], y)
                    )
                    loss = loss + loss_perturb
                    logs["loss_perturb"] += loss_perturb.detach()

            if (
                use_clip
                and outputs["z_body"] is not None
                and outputs["x_body"] is not None
            ):
                loss_clip = clip_loss(
                    outputs["x_body"].view(X.shape[0], -1),
                    outputs["z_body"].view(X.shape[0], -1),
                    weight=outputs["alpha"],
                )
                loss = loss + loss_clip
                logs["loss_clip"] += loss_clip.detach()

        logs["loss"] += loss.detach()
        if use_amp and gscaler is not None:
            gscaler.scale(loss).backward()
            gscaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            gscaler.step(optimizer)
            gscaler.update()
        else:
            loss.backward()  # Backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # Update parameters
        scheduler.step()

    if dist.is_initialized():
        for key in logs:
            dist.all_reduce(logs[key].detach())
            logs[key] = float(logs[key] / dist.get_world_size() / iterations_per_epoch)

    return logs


def test_step(
    model,
    dataloader,
    class_cost,
    gen_cost,
    epoch,
    device,
    clip_loss=CLIPLoss(),
    use_clip=False,
    use_event_loss=False,
    iterations_per_epoch=-1,
):
    model.eval()

    logs_buff = torch.zeros((7), dtype=torch.float32, device=device)
    logs = {}
    logs["loss"] = logs_buff[0].view(-1)
    logs["loss_class"] = logs_buff[1].view(-1)
    logs["loss_gen"] = logs_buff[2].view(-1)
    logs["loss_perturb"] = logs_buff[3].view(-1)
    logs["loss_clip"] = logs_buff[4].view(-1)
    logs["loss_class_event"] = logs_buff[5].view(-1)
    logs["loss_event_perturb"] = logs_buff[6].view(-1)

    if iterations_per_epoch < 0:
        iterations_per_epoch = len(dataloader)

    data_iter = iter(dataloader)
    
    for batch_idx in range(iterations_per_epoch):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # for batch_idx, batch in enumerate(dataloader):
        # X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        # # y in G4 dataset is hot_encoded, but here it takes int categories.
        # y = torch.argmax(y, dim=1)
        X, energy, y, gap_pid = batch
        X, energy, y, gap_pid = X.to(device), energy.to(device), y.to(device), gap_pid.to(device)
        y = (y == 2).long()
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }
        try:
            with torch.no_grad():
                outputs = model(X, y, gap_pid, energy, **model_kwargs)
        except Exception as e:
            print(f"batch_idx [{batch_idx}] skiped: Exception during model inference: {e}")
            continue
        loss = 0

        if outputs["y_pred"] is not None:
            if use_event_loss:
                event_mask = y >= 200
                if event_mask.any():
                    loss_event = class_cost(
                        outputs["y_pred"][event_mask][:, 200:], y[event_mask] - 200
                    ).mean()
                    logs["loss_class_event"] += loss_event.detach()
                    loss = loss + loss_event
                if (~event_mask).any():
                    loss_class = class_cost(
                        outputs["y_pred"][~event_mask][:, :200], y[~event_mask]
                    ).mean()
                    logs["loss_class"] += loss_class.detach()
                    loss = loss + loss_class
            else:
                loss_class = class_cost(outputs["y_pred"], y).mean()
                #FIXME
                #y_int_labels = torch.argmax(y, dim=1)
                #loss_class = class_cost(outputs["y_pred"], y_int_labels).mean()
                loss = loss + loss_class
                logs["loss_class"] += loss_class.detach()
        if outputs["z_pred"] is not None:
            nonzero = (outputs["v"][:, :, 0] != 0).sum(1)
            loss_gen = gen_cost(outputs["v"], outputs["z_pred"]).sum((1, 2)) / nonzero
            loss_gen = loss_gen.mean()
            loss = loss + loss_gen
            logs["loss_gen"] += loss_gen.detach()
        if outputs["y_perturb"] is not None:
            if use_event_loss:
                event_mask = y >= 200
                if event_mask.any():
                    loss_event = torch.mean(
                        outputs["alpha"][event_mask].squeeze(1)
                        * class_cost(
                            outputs["y_perturb"][event_mask][:, 200:],
                            y[event_mask] - 200,
                        )
                    )
                    logs["loss_event_perturb"] += loss_event.detach()
                    loss = loss + loss_event
                if (~event_mask).any():
                    loss_class = torch.mean(
                        outputs["alpha"][~event_mask].squeeze(1)
                        * class_cost(
                            outputs["y_perturb"][~event_mask][:, :200], y[~event_mask]
                        )
                    )
                    logs["loss_perturb"] += loss_class.detach()
                    loss = loss + loss_class

            else:
                loss_perturb = torch.mean(
                    outputs["alpha"].squeeze(1) * class_cost(outputs["y_perturb"], y)
                )
                loss = loss + loss_perturb
                logs["loss_perturb"] += loss_perturb.detach()

        if use_clip and outputs["z_body"] is not None and outputs["x_body"] is not None:
            loss_clip = clip_loss(
                outputs["x_body"].view(X.shape[0], -1),
                outputs["z_body"].view(X.shape[0], -1),
                weight=outputs["alpha"],
            )
            loss = loss + loss_clip
            logs["loss_clip"] += loss_clip.detach()

        #print(f"################3 loss {loss.detach()}")
        logs["loss"] += loss.detach()

    if dist.is_initialized():
        for key in logs:
            dist.all_reduce(logs[key].detach())
            logs[key] = float(logs[key] / dist.get_world_size() / iterations_per_epoch)

    return logs


def train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    lr_scheduler,
    num_epochs=1,
    device="cpu",
    patience=500,
    loss_class=nn.CrossEntropyLoss(),
    loss_gen=nn.MSELoss(),
    use_clip=False,
    use_event_loss=False,
    output_dir="",
    save_tag="",
    iterations_per_epoch=-1,
    epoch_init=0,
    loss_init=np.inf,
    use_amp=False,
    run=None,
):
    checkpoint_name = get_checkpoint_name(save_tag)

    losses = {
        "train_loss": [],
        "val_loss": [],
    }

    tracker = {"bestValLoss": loss_init, "bestEpoch": epoch_init}
    if use_amp:
        gscaler = amp.GradScaler()
    else:
        gscaler = None
    for epoch in range(int(epoch_init), num_epochs):
        if isinstance(
            train_loader.sampler, torch.utils.data.distributed.DistributedSampler
        ):
            train_loader.sampler.set_epoch(epoch)

        start = time.time()
        train_logs = train_step(
            model,
            train_loader,
            loss_class,
            loss_gen,
            optimizer,
            lr_scheduler,
            epoch,
            device,
            use_clip=use_clip,
            use_event_loss=use_event_loss,
            iterations_per_epoch=iterations_per_epoch,
            use_amp=use_amp,
            gscaler=gscaler,
        )
        val_logs = test_step(
            model,
            test_loader,
            loss_class,
            loss_gen,
            epoch,
            device,
            use_clip=use_clip,
            use_event_loss=use_event_loss,
            iterations_per_epoch=iterations_per_epoch,
        )

        losses["train_loss"].append(train_logs["loss"])
        losses["val_loss"].append(val_logs["loss"])

        if is_master_node():
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] Loss: {losses['train_loss'][-1]:.4f}, Val Loss: {losses['val_loss'][-1]:.4f} , lr: {lr_scheduler.get_last_lr()[0]}"
                f"Epoch [{epoch + 1}/{num_epochs}] Loss: {losses['train_loss'][-1]:.4f}, Val Loss: {losses['val_loss'][-1]:.4f} , lr: {lr_scheduler.get_last_lr()[0]}"
            )
            print(
                f"Class Loss: {train_logs['loss_class']:.4f}, Class Val Loss: {val_logs['loss_class']:.4f}"
            )
            print(
                f"Class Event Loss: {train_logs['loss_class_event']:.4f}, Class Event Val Loss: {val_logs['loss_class_event']:.4f}"
            )
            print(
                f"Gen Loss: {train_logs['loss_gen']:.4f}, Gen Val Loss: {val_logs['loss_gen']:.4f}"
            )
            print(
                f"Class Perturb Loss: {train_logs['loss_perturb']:.4f}, Class Val Perturb Loss: {val_logs['loss_perturb']:.4f}"
            )
            print(
                f"CLIP loss: {train_logs['loss_clip']:.4f}, CLIP Val Loss: {val_logs['loss_clip']:.4f}"
            )
            print(
                "Time taken for epoch {} is {} sec".format(epoch, time.time() - start)
            )

        if losses["val_loss"][-1] < tracker["bestValLoss"]:
            tracker["bestValLoss"] = losses["val_loss"][-1]
            tracker["bestEpoch"] = epoch

        if is_master_node():
            print("replacing best checkpoint ...")
            save_checkpoint(
                model,
                epoch + 1,
                optimizer,
                losses["val_loss"][-1],
                lr_scheduler,
                output_dir,
                checkpoint_name,
            )

        if run is not None:
            for key in train_logs:
                run.log({f"train {key}": train_logs[key]})
            for key in val_logs:
                run.log({f"val {key}": val_logs[key]})

        if epoch - tracker["bestEpoch"] > patience:
            print(f"breaking on device: {device}")
            break

    if is_master_node():
        print(
            f"Training Complete, best loss: {tracker['bestValLoss']:.5f} at epoch {tracker['bestEpoch']}!"
        )
        # save losses
        json.dump(losses, open(f"{output_dir}/training_{save_tag}.json", "w"))


def save_checkpoint(
    model, epoch, optimizer, loss, lr_scheduler, checkpoint_dir, checkpoint_name
):
    save_dict = {
        "body": model.module.body.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "sched": lr_scheduler.state_dict(),
    }

    if model.module.classifier is not None:
        save_dict["classifier_head"] = model.module.classifier.state_dict()

    if model.module.generator is not None:
        save_dict["generator_head"] = model.module.generator.state_dict()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.save(save_dict, os.path.join(checkpoint_dir, checkpoint_name))
    print(
        f"Epoch {epoch} | Training checkpoint saved at {os.path.join(checkpoint_dir, checkpoint_name)}"
    )


def restore_checkpoint(
    model,
    optimizer,
    lr_scheduler,
    checkpoint_dir,
    checkpoint_name,
    device,
    is_main_node=False,
    fine_tune=False,
):
    device = "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, checkpoint_name),
        map_location=device,
    )

    base_model = model.module if hasattr(model, "module") else model
    base_model.to(device)
    base_model.body.load_state_dict(checkpoint["body"], strict=False)

    if not fine_tune:
        if base_model.classifier is not None and "classifier_head" in checkpoint:
            base_model.classifier.load_state_dict(
                checkpoint["classifier_head"], strict=False
            )

        if base_model.generator is not None:
            base_model.generator.load_state_dict(
                checkpoint["generator_head"], strict=False
            )

        lr_scheduler.load_state_dict(checkpoint["sched"])
        startEpoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"]
    else:
        if base_model.classifier is not None and "classifier_head" in checkpoint:
            classifier_state = checkpoint["classifier_head"]
            model_state = base_model.classifier.state_dict()
            filtered_state = {}
            for k, v in classifier_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    if is_main_node:
                        print(
                            f"Skipping {k}: shape mismatch (checkpoint: {v.shape}, model: {model_state[k].shape if k in model_state else 'missing'})"
                        )

            base_model.classifier.load_state_dict(filtered_state, strict=False)

        if base_model.generator is not None:
            classifier_state = checkpoint["generator_head"]
            model_state = base_model.generator.state_dict()
            filtered_state = {}
            for k, v in classifier_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    if is_main_node:
                        print(
                            f"Skipping {k}: shape mismatch (checkpoint: {v.shape}, model: {model_state[k].shape if k in model_state else 'missing'})"
                        )

            base_model.generator.load_state_dict(filtered_state, strict=False)

        startEpoch = 0.0
        best_loss = np.inf

    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except Exception:
        if is_main_node:
            print("Optimizer cannot be loaded back, skipping...")

    return startEpoch, best_loss


def main(args):
    local_rank, rank, size = ddp_setup()
    # set up model
    model = PET2(
        input_dim=args.num_feat,
        hidden_size=args.base_dim,
        num_transformers=args.num_transf,
        num_transformers_head=args.num_transf_heads,
        num_heads=args.num_head,
        mlp_ratio=args.mlp_ratio,
        mlp_drop=args.mlp_drop,
        attn_drop=args.attn_drop,
        feature_drop=args.feature_drop,
        num_tokens=args.num_tokens,
        K=args.K,
        use_int=args.use_interaction,
        conditional=args.conditional,
        cond_dim=args.num_cond,
        pid=args.use_pid,
        add_info=args.use_add,
        add_dim=args.num_add,
        use_time=False if args.mode == "classifier" else True,
        mode=args.mode,
        num_classes=args.num_classes,
        num_gap_classes=args.num_gap_classes,
    )
    if rank == 0:
        d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("**** Setup ****")
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )
        print(f"Training on device: {d}, with {size} GPUs")
        print("************")

    # load in train data
    # train_loader = load_data(
    #     args.dataset,
    #     dataset_type="train",
    #     use_pid=args.use_pid,
    #     pid_idx=args.pid_idx,
    #     use_add=args.use_add,
    #     num_add=args.num_add,
    #     path=args.path,
    #     batch=args.batch,
    #     num_workers=args.num_workers,
    #     rank=rank,
    #     size=size,
    # )
    # if rank == 0:
    #     print("**** Setup ****")
    #     print(f"Train dataset len: {len(train_loader)}")
    #     print("************")

    # test_loader = load_data(
    #     args.dataset,
    #     dataset_type="test",
    #     use_pid=args.use_pid,
    #     pid_idx=args.pid_idx,
    #     use_add=args.use_add,
    #     num_add=args.num_add,
    #     path=args.path,
    #     batch=args.batch,
    #     num_workers=args.num_workers,
    #     rank=rank,
    #     size=size,
    # )
    #TODO unify for all datasets,. Better make a class that load the dataset and split it
    if args.dataset == "G4":#pkl
        pkl_files_path = args.path
        dataset = PklDataset(pkl_files_path)
    elif args.dataset == "G4_h5":#h5
        h5_file_path = args.path
        dataset = HDF5Dataset(h5_file_path)
    elif args.dataset == "ShapeNetCore":#shapenetcore
        shapenet_path = args.path
        dataset = ShapeNetCore(shapenet_path)

    print(f"Successfully loaded dataset with {len(dataset)} total events.")
        
    # Define the split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # Calculate the number of samples for each split
    num_events = len(dataset)
    num_train = int(num_events * train_ratio)
    num_val = int(num_events * val_ratio)
    num_test = num_events - num_train - num_val

    # Use random_split to create the subsets
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [num_train, num_val, num_test]
    )

    print(f"\nDataset split into:")
    print(f"  Training set: {len(train_dataset)} events")
    print(f"  Validation set: {len(val_dataset)} events")
    print(f"  Test set: {len(test_dataset)} events")

    # Create DataLoaders for each subset
    batch_size = args.batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)#, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)#, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)#, num_workers=args.num_workers)

    
    if rank == 0:
        print("**** Setup ****")
        print(f"Train dataset len: {len(train_loader)}")
        print("************")

    param_groups = get_param_groups(
        model, args.wd, args.lr, lr_factor=args.lr_factor, fine_tune=args.fine_tune
    )

    if args.optim not in ["adamw", "lion"]:
        raise ValueError(
            f"Optimizer '{args.optim}' not supported. Choose from adam or lion."
        )

    # if args.optim == "lion":
    #     optimizer = Lion(param_groups, betas=(args.b1, args.b2))
    # if args.optim == "adam":
    optimizer = torch.optim.AdamW(param_groups)

    train_steps = len(train_loader) if args.iterations < 0 else args.iterations

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, (train_steps * epoch)
    # )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=train_steps * args.warmup_epoch,
        num_training_steps=(train_steps * args.epoch),
    )

    epoch_init = 0
    loss_init = np.inf

    if os.path.isfile(os.path.join(args.outdir, get_checkpoint_name(args.save_tag))) and args.resuming:
        if is_master_node():
            print(
                f"Continue training with checkpoint from {os.path.join(args.outdir, get_checkpoint_name(args.save_tag))}"
            )

        epoch_init, loss_init = restore_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            args.outdir,
            get_checkpoint_name(args.save_tag),
            local_rank,
        )

    if (
        os.path.isfile(os.path.join(args.outdir, get_checkpoint_name(args.pretrain_tag)))
        and args.fine_tune
    ):
        if is_master_node():
            print(
                f"Will fine-tune using checkpoint {os.path.join(args.outdir, get_checkpoint_name(args.pretrain_tag))}"
            )

        epoch_init, loss_init = restore_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            args.outdir,
            get_checkpoint_name(args.pretrain_tag),
            local_rank,
            is_main_node=is_master_node(),
            fine_tune=args.fine_tune,
        )

    # Transfer model to GPU if available
    kwarg = {}
    if torch.cuda.is_available():
        device = local_rank
        model.to(local_rank)
        kwarg["device_ids"] = [device]
    else:
        model.cpu()
        device = "cpu"

    model = DDP(
        model,
        **kwarg,
    )

    if args.wandb:
        import wandb

        if is_master_node():
            mode_wandb = None
            wandb.login()
        else:
            mode_wandb = "disabled"

        run = wandb.init(
            # Set the project where this run will be logged
            project="OmniLearn",
            name=args.save_tag,
            mode=mode_wandb,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "epochs": args.epoch,
                "batch size": args.batch,
                "mode": args.mode,
            },
        )
    else:
        run = None

    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        num_epochs=args.epoch,
        device=device,
        loss_class=nn.CrossEntropyLoss(reduction="none"),
        loss_gen=nn.MSELoss(reduction="none"),
        output_dir=args.outdir,
        save_tag=args.save_tag,
        use_clip=args.use_clip,
        use_event_loss=args.use_event_loss,
        iterations_per_epoch=args.iterations,
        epoch_init=epoch_init,
        loss_init=loss_init,
        use_amp=args.use_amp,
        run=run,
    )

    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)