# Copyright 2025 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

from typing import Optional

from ..cluster import ClusterConfig, NodeGroupInfo, NodeInfo
from .manager import Manager


class NodeManager(Manager):
    """Global manager of node metadata information."""

    MANAGER_NAME = "NodeManager"

    def __init__(
        self,
        nodes: list[NodeInfo],
        node_groups: list[NodeGroupInfo],
        cluster_cfg: Optional[ClusterConfig],
    ):
        """Initialize the NodeManager.

        Args:
            nodes (list[NodeInfo]): List of NodeInfo objects representing the nodes in the cluster
            node_groups (list[NodeGroupInfo]): List of NodeGroupInfo objects representing the node groups in the cluster
            cluster_cfg (Optional[ClusterConfig]): Configuration of the cluster.

        """
        self._nodes = nodes
        self._node_groups = node_groups
        self._cluster_cfg = cluster_cfg

    def get_nodes(self):
        """Get the list of nodes in the cluster."""
        return self._nodes, self._node_groups, self._cluster_cfg
