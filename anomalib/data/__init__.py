"""Anomalib Datasets."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from omegaconf import DictConfig, ListConfig

from .base import AnomalibDataModule, AnomalibDataset
from .inference import InferenceDataset
from .task_type import TaskType
from .one_dimension_packet import One_dimension_packet


logger = logging.getLogger(__name__)


def get_datamodule(config: DictConfig | ListConfig) -> AnomalibDataModule:
    """Get Anomaly Datamodule.

    Args:
        config (DictConfig | ListConfig): Configuration of the anomaly model.

    Returns:
        PyTorch Lightning DataModule
    """
    logger.info("Loading the datamodule")

    datamodule: AnomalibDataModule

    # convert center crop to tuple
    center_crop = config.dataset.get("center_crop")
    if center_crop is not None:
        center_crop = (center_crop[0], center_crop[1])

    if config.dataset.format.lower() == "one_dimension_packet":
        #######传入参数
        datamodule = One_dimension_packet(
            root=config.dataset.path,
            normal_path=config.dataset.normal_dir,
            abnormal_path=config.dataset.abnormal_dir,
            normal_test_path=config.dataset.normal_test_dir,
            split_ratio=config.dataset.split_ratio,   # 用来创造验证集的比例
            create_validtion_set=config.dataset.create_validation_set,
            train_batch_size=config.dataset.train_batch_size,
            inference_batch_size=config.dataset.inference_batch_size,
            num_workers=config.dataset.num_workers,
        )
    else:
        raise ValueError(
            "Unknown dataset! \n"
            "If you use a custom dataset make sure you initialize it in"
            "`get_datamodule` in `anomalib.data.__init__.py"
        )

    return datamodule


__all__ = [
    "AnomalibDataset",
    "AnomalibDataModule",
    "get_datamodule",
    "InferenceDataset",
    "TaskType",
    "One_dimension_packet"
]
