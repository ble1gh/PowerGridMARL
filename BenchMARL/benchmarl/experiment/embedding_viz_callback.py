"""Callback that visualizes learned GNN embeddings during training.

Generates three plots every ``log_every_n_iters`` collection iterations:
  1. **t-SNE scatter** — 2-D non-linear projection colored by agent group.
  2. **PCA scatter** — 2-D linear projection (stable axes across steps).
  3. **Cosine-similarity heatmap** — N×N pairwise similarity, sorted by group.

All plots are logged to wandb under the ``embeddings/`` key prefix.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordict import TensorDictBase

from benchmarl.experiment.callback import Callback


class EmbeddingVizCallback(Callback):
    """Log t-SNE, PCA, and cosine-similarity visualizations of ``embedding_z``.

    Args:
        log_every_n_iters: how often (in collection iterations) to generate
            plots.  Defaults to 50.
        max_samples: maximum number of agent-timestep samples to include per
            group.  Embeddings are randomly sub-sampled to this limit to keep
            t-SNE fast.  Defaults to 512.
        embedding_key: the tensordict leaf key holding the embedding.
            Defaults to ``"embedding_z"``.
        tsne_perplexity: perplexity parameter for t-SNE.  Defaults to 30.
    """

    def __init__(
        self,
        log_every_n_iters: int = 50,
        max_samples: int = 512,
        embedding_key: str = "embedding_z",
        tsne_perplexity: float = 30.0,
    ):
        super().__init__()
        self.log_every_n_iters = log_every_n_iters
        self.max_samples = max_samples
        self.embedding_key = embedding_key
        self.tsne_perplexity = tsne_perplexity
        self._pca_fit = None  # will be fit once and reused for stable axes

    # ------------------------------------------------------------------
    # Callback entry point
    # ------------------------------------------------------------------

    def on_batch_collected(self, batch: TensorDictBase):
        exp = self.experiment
        if exp.n_iters_performed % self.log_every_n_iters != 0:
            return

        # Collect embeddings from all groups
        embeddings, labels, group_order = self._extract_embeddings(batch)
        if embeddings is None:
            return  # no embeddings found (gnn_mode == "none")

        # Build figures
        fig_tsne = self._plot_tsne(embeddings, labels, group_order)
        fig_pca = self._plot_pca(embeddings, labels, group_order)
        fig_heatmap = self._plot_cosine_heatmap(embeddings, labels, group_order)

        # Log to wandb
        step = exp.n_iters_performed
        self._log_figure("embeddings/tsne", fig_tsne, step)
        self._log_figure("embeddings/pca", fig_pca, step)
        self._log_figure("embeddings/cosine_similarity", fig_heatmap, step)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_embeddings(self, batch: TensorDictBase):
        """Pull ``embedding_z`` from each group, flatten, sub-sample.

        Returns:
            embeddings: np.ndarray of shape (N, z_dim)
            labels: np.ndarray of shape (N,) — integer group index per sample
            group_order: list of group name strings
        """
        group_map = self.experiment.group_map
        all_emb = []
        all_labels = []
        group_order = list(group_map.keys())

        for g_idx, group in enumerate(group_order):
            z = batch.get((group, self.embedding_key), None)
            if z is None:
                return None, None, None
            # z shape: [n_envs, traj_len, n_agents, z_dim] or sub-shapes
            z_flat = z.detach().cpu().float().reshape(-1, z.shape[-1])
            # Sub-sample if too many points
            if z_flat.shape[0] > self.max_samples:
                idx = torch.randperm(z_flat.shape[0])[: self.max_samples]
                z_flat = z_flat[idx]
            all_emb.append(z_flat.numpy())
            all_labels.append(np.full(z_flat.shape[0], g_idx, dtype=np.int64))

        embeddings = np.concatenate(all_emb, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return embeddings, labels, group_order

    # ------------------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------------------

    def _plot_tsne(self, embeddings, labels, group_order):
        from sklearn.manifold import TSNE

        perplexity = min(self.tsne_perplexity, embeddings.shape[0] - 1)
        proj = TSNE(
            n_components=2, perplexity=perplexity, random_state=42, init="pca"
        ).fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = plt.cm.get_cmap("tab10", len(group_order))
        for g_idx, name in enumerate(group_order):
            mask = labels == g_idx
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=[cmap(g_idx)],
                label=name,
                alpha=0.6,
                s=12,
                edgecolors="none",
            )
        ax.legend(fontsize=8)
        ax.set_title(f"t-SNE  (iter {self.experiment.n_iters_performed})")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    def _plot_pca(self, embeddings, labels, group_order):
        from sklearn.decomposition import PCA

        if self._pca_fit is None:
            self._pca_fit = PCA(n_components=2, random_state=42)
            self._pca_fit.fit(embeddings)

        proj = self._pca_fit.transform(embeddings)
        var = self._pca_fit.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = plt.cm.get_cmap("tab10", len(group_order))
        for g_idx, name in enumerate(group_order):
            mask = labels == g_idx
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=[cmap(g_idx)],
                label=name,
                alpha=0.6,
                s=12,
                edgecolors="none",
            )
        ax.legend(fontsize=8)
        ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
        ax.set_title(f"PCA  (iter {self.experiment.n_iters_performed})")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Cosine-similarity heatmap
    # ------------------------------------------------------------------

    def _plot_cosine_heatmap(self, embeddings, labels, group_order):
        # Sort by group so blocks are contiguous
        order = np.argsort(labels, kind="stable")
        sorted_emb = embeddings[order]
        sorted_labels = labels[order]

        # Cosine similarity
        norms = np.linalg.norm(sorted_emb, axis=1, keepdims=True) + 1e-8
        normed = sorted_emb / norms
        sim = normed @ normed.T

        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Draw group boundary lines and labels
        boundaries = []
        for g_idx, name in enumerate(group_order):
            count = (sorted_labels == g_idx).sum()
            start = boundaries[-1] if boundaries else 0
            end = start + count
            boundaries.append(end)
            mid = (start + end) / 2
            ax.text(
                -0.5,
                mid,
                name,
                ha="right",
                va="center",
                fontsize=7,
                transform=ax.get_yaxis_transform(),
            )
            if g_idx > 0:
                ax.axhline(start - 0.5, color="k", linewidth=0.5)
                ax.axvline(start - 0.5, color="k", linewidth=0.5)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Cosine Similarity  (iter {self.experiment.n_iters_performed})")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log_figure(self, key: str, fig, step: int):
        """Log a matplotlib figure to wandb (if active)."""
        from torchrl.record.loggers.wandb import WandbLogger

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        buf.seek(0)

        for logger in self.experiment.logger.loggers:
            if isinstance(logger, WandbLogger):
                import wandb
                from PIL import Image

                img = Image.open(buf)
                buf.seek(0)
                logger.experiment.log({key: wandb.Image(img)}, commit=False)

        buf.close()
        plt.close(fig)
