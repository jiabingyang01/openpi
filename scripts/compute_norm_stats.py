"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _fast_parquet_norm_stats(local_roots, action_dim=14, max_frames=None):
    """Compute norm stats directly from parquet files, skipping video decoding."""
    import pandas as pd
    import pathlib

    if isinstance(local_roots, str):
        local_roots = [local_roots]

    all_states = []
    all_actions = []
    total = 0

    for root in local_roots:
        root = pathlib.Path(root)
        parquets = sorted(root.glob("data/chunk-*/episode_*.parquet"))
        for pf in parquets:
            df = pd.read_parquet(pf, columns=["observation.state", "action"])
            states = np.stack(df["observation.state"].tolist())[:, :action_dim]
            actions = np.stack(df["action"].tolist())[:, :action_dim]
            all_states.append(states)
            all_actions.append(actions)
            total += len(states)
            if max_frames and total >= max_frames:
                break
        if max_frames and total >= max_frames:
            break

    all_states = np.concatenate(all_states)[:max_frames] if max_frames else np.concatenate(all_states)
    all_actions = np.concatenate(all_actions)[:max_frames] if max_frames else np.concatenate(all_actions)

    stats = {
        "state": normalize.NormStats(
            mean=all_states.mean(axis=0),
            std=all_states.std(axis=0),
            q01=np.percentile(all_states, 1, axis=0),
            q99=np.percentile(all_states, 99, axis=0),
        ),
        "actions": normalize.NormStats(
            mean=all_actions.mean(axis=0),
            std=all_actions.std(axis=0),
            q01=np.percentile(all_actions, 1, axis=0),
            q99=np.percentile(all_actions, 99, axis=0),
        ),
    }
    print(f"Fast parquet stats: {total} frames from {len(local_roots)} roots")
    return stats


def _fast_parquet_norm_stats_with_vae(local_roots, vae_encoder, max_frames=None):
    """Compute norm stats in VAE latent space: read parquet → VAE encode → stats."""
    import pandas as pd
    import pathlib

    if isinstance(local_roots, str):
        local_roots = [local_roots]

    all_states = []
    all_latents = []
    total = 0

    for root in local_roots:
        root = pathlib.Path(root)
        parquets = sorted(root.glob("data/chunk-*/episode_*.parquet"))
        for pf in parquets:
            df = pd.read_parquet(pf, columns=["observation.state", "action"])
            states = np.stack(df["observation.state"].tolist())[:, :14]
            actions = np.stack(df["action"].tolist())[:, :14]
            all_states.append(states)

            # Encode each action through VAE (process per-frame, VAE expects chunks)
            # Just store raw actions, we'll encode in batch below
            all_latents.append(actions)
            total += len(states)
            if max_frames and total >= max_frames:
                break
        if max_frames and total >= max_frames:
            break

    all_states = np.concatenate(all_states)[:max_frames] if max_frames else np.concatenate(all_states)
    raw_actions = np.concatenate(all_latents)[:max_frames] if max_frames else np.concatenate(all_latents)

    # Encode actions through frozen VAE in batches
    # The VAE encoder expects (action_horizon, 14) per sample, so we feed individual frames
    # wrapped as the data transform would
    print(f"Encoding {len(raw_actions)} action frames through frozen VAE...")
    encoded = []
    for i in tqdm.tqdm(range(len(raw_actions)), desc="VAE encoding"):
        sample = {"actions": raw_actions[i:i+1].astype(np.float32)}  # (1, 14)
        result = vae_encoder(sample)
        encoded.append(result["actions"])  # latent shape

    all_actions_latent = np.concatenate(encoded, axis=0) if encoded[0].ndim > 1 else np.stack(encoded)

    stats = {
        "state": normalize.NormStats(
            mean=all_states.mean(axis=0),
            std=all_states.std(axis=0),
            q01=np.percentile(all_states, 1, axis=0),
            q99=np.percentile(all_states, 99, axis=0),
        ),
        "actions": normalize.NormStats(
            mean=all_actions_latent.mean(axis=0),
            std=all_actions_latent.std(axis=0),
            q01=np.percentile(all_actions_latent, 1, axis=0),
            q99=np.percentile(all_actions_latent, 99, axis=0),
        ),
    }
    print(f"Fast parquet+VAE stats: {total} frames, latent shape={all_actions_latent.shape[1:]}")
    return stats


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # Fast path: if local_root is available, read parquets directly (no video decode).
    local_root = data_config.local_root
    # Check if DynaActVAE encoder is in the data transforms — if so, need to encode actions.
    vae_encoder = None
    for t in data_config.data_transforms.inputs:
        if hasattr(t, 'checkpoint_path') and hasattr(t, 'z_dim') and t.checkpoint_path:
            vae_encoder = t
            break
    if local_root is not None and vae_encoder is None:
        norm_stats = _fast_parquet_norm_stats(local_root, max_frames=max_frames)
        output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
        return
    if local_root is not None and vae_encoder is not None:
        # Fast path + VAE: read parquet, encode through frozen VAE, compute latent stats.
        norm_stats = _fast_parquet_norm_stats_with_vae(local_root, vae_encoder, max_frames=max_frames)
        output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
        return

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
