from __future__ import annotations

from dataclasses import dataclass


# A healthy snapshot may contain many tiny lineage records, but it must not turn the
# per-artifact limits into hundreds of megabytes or millions of retained JSON nodes.
MAX_DOMAIN_SNAPSHOT_BYTES = 16 * 1024 * 1024
MAX_DOMAIN_SNAPSHOT_JSON_NODES = 256 * 1024


@dataclass
class DomainSnapshotBudget:
    decoded_nodes: int = 0

    def require_total_bytes(self, total_bytes: int) -> None:
        if total_bytes < 0 or total_bytes > MAX_DOMAIN_SNAPSHOT_BYTES:
            raise ValueError("domain_snapshot_bytes_exceeded")

    def consume_json(self, value: object) -> None:
        self.consume_nodes(_json_node_count(value))

    def consume_nodes(self, nodes: int) -> None:
        if nodes < 0 or self.decoded_nodes + nodes > MAX_DOMAIN_SNAPSHOT_JSON_NODES:
            raise ValueError("domain_snapshot_json_nodes_exceeded")
        self.decoded_nodes += nodes


def _json_node_count(value: object) -> int:
    nodes = 0
    stack = [value]
    while stack:
        current = stack.pop()
        nodes += 1
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return nodes
