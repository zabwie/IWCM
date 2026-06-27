"""Grid world objects and inventory system (Section 7.1).

Defines object types (Key, Door, Box, Occluder), their properties,
and an inventory system for holding carried objects.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
from copy import deepcopy

# Re-exported from grid_world.py for clean imports
from .grid_world import OBJECT_TYPES


# ═══════════════════════════════════════════════════════════
# Object Dataclasses
# ═══════════════════════════════════════════════════════════

@dataclass
class GridObject:
    """Base class for all grid world objects."""
    obj_id: str
    obj_type: str
    pos: Tuple[int, int]
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def row(self) -> int:
        return self.pos[0]

    @property
    def col(self) -> int:
        return self.pos[1]

    @property
    def is_passable(self) -> bool:
        return self.properties.get("passable", False)

    @property
    def is_pickupable(self) -> bool:
        return self.properties.get("pickupable", False)

    @property
    def is_pushable(self) -> bool:
        return self.properties.get("pushable", False)

    def to_dict(self) -> dict:
        return {
            "type": self.obj_type,
            "pos": list(self.pos),
            "properties": deepcopy(self.properties),
        }

    @classmethod
    def from_dict(cls, obj_id: str, d: dict) -> "GridObject":
        return cls(
            obj_id=obj_id,
            obj_type=d["type"],
            pos=tuple(d["pos"]),
            properties=deepcopy(d.get("properties", {})),
        )


@dataclass
class Key(GridObject):
    """A key object that can be picked up and used to open matching doors."""
    obj_type: str = "key"

    @classmethod
    def create(cls, obj_id: str, pos: Tuple[int, int]) -> "Key":
        return cls(
            obj_id=obj_id,
            obj_type="key",
            pos=pos,
            properties={
                "pickupable": True,
                "pushable": False,
                "passable": True,
                "color": "yellow",
            },
        )


@dataclass
class Door(GridObject):
    """A door that blocks passage when closed, can be opened with matching key."""
    obj_type: str = "door"
    key_id: Optional[str] = None
    is_open: bool = False

    @classmethod
    def create(
        cls, obj_id: str, pos: Tuple[int, int], key_id: str
    ) -> "Door":
        return cls(
            obj_id=obj_id,
            obj_type="door",
            pos=pos,
            key_id=key_id,
            properties={
                "pickupable": False,
                "pushable": False,
                "passable": True,
                "color": "red",
                "openable": True,
            },
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["key_id"] = self.key_id
        return d


@dataclass
class Box(GridObject):
    """A pushable box that blocks movement but can be pushed by the agent."""
    obj_type: str = "box"

    @classmethod
    def create(cls, obj_id: str, pos: Tuple[int, int]) -> "Box":
        return cls(
            obj_id=obj_id,
            obj_type="box",
            pos=pos,
            properties={
                "pickupable": False,
                "pushable": True,
                "passable": False,
                "color": "brown",
            },
        )


@dataclass
class Occluder(GridObject):
    """A static obstacle that blocks both movement and line of sight."""
    obj_type: str = "occluder"

    @classmethod
    def create(cls, obj_id: str, pos: Tuple[int, int]) -> "Occluder":
        return cls(
            obj_id=obj_id,
            obj_type="occluder",
            pos=pos,
            properties={
                "pickupable": False,
                "pushable": False,
                "passable": False,
                "color": "gray",
            },
        )


# ═══════════════════════════════════════════════════════════
# Object Factory
# ═══════════════════════════════════════════════════════════

_OBJECT_CLASSES: Dict[str, type] = {
    "key": Key,
    "door": Door,
    "box": Box,
    "occluder": Occluder,
}


def create_object(
    obj_type: str, obj_id: str, pos: Tuple[int, int], **kwargs
) -> GridObject:
    """Factory function to create grid objects by type.

    Args:
        obj_type: One of "key", "door", "box", "occluder".
        obj_id: Unique identifier for the object.
        pos: (row, col) position on the grid.
        **kwargs: Type-specific parameters (e.g., key_id for doors).

    Returns:
        A GridObject subclass instance.
    """
    cls = _OBJECT_CLASSES.get(obj_type)
    if cls is None:
        raise ValueError(f"Unknown object type: {obj_type}")

    if obj_type == "door":
        return cls.create(obj_id=obj_id, pos=pos, key_id=kwargs.get("key_id"))
    else:
        return cls.create(obj_id=obj_id, pos=pos)


# ═══════════════════════════════════════════════════════════
# Inventory System
# ═══════════════════════════════════════════════════════════

class Inventory:
    """Tracks objects carried by the agent.

    Supports add, remove, contains check, and iteration.
    """

    def __init__(self, capacity: int = 4):
        self._items: list[str] = []  # list of object_ids
        self.capacity = capacity

    def add(self, obj_id: str) -> bool:
        """Add object to inventory if not full. Returns True on success."""
        if self.is_full:
            return False
        if obj_id in self._items:
            return False
        self._items.append(obj_id)
        return True

    def remove(self, obj_id: str) -> Optional[str]:
        """Remove and return object_id if present, else None."""
        if obj_id in self._items:
            self._items.remove(obj_id)
            return obj_id
        return None

    def pop_last(self) -> Optional[str]:
        """Remove and return the most recently added item."""
        if self._items:
            return self._items.pop()
        return None

    def contains(self, obj_id: str) -> bool:
        return obj_id in self._items

    def contains_any(self, obj_ids: list) -> bool:
        return any(o in self._items for o in obj_ids)

    @property
    def is_full(self) -> bool:
        return len(self._items) >= self.capacity

    @property
    def is_empty(self) -> bool:
        return len(self._items) == 0

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, obj_id: str) -> bool:
        return obj_id in self._items

    def __iter__(self):
        return iter(self._items)

    def to_list(self) -> list:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()

    def __repr__(self) -> str:
        return f"Inventory({self._items})"
