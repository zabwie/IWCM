"""Encoder — Video encoder, slot attention, spatial anchoring, slot permanence, decoder."""

from .video_encoder import VideoEncoder
from .slot_attention import SlotAttention
from .slot_transition import SlotTransitionPredictor
from .slot_permanence import (
    SlotPermanenceEncoder,
    content_smoothness_loss,
    slot_diversity_loss,
    transition_consistency_loss,
    compute_slot_switch_rate,
)
from .spatial_anchor import (
    build_position_grid,
    init_anchor_grid,
    compute_spatial_attention_bias,
)
from .decoder import VideoDecoder
from .representation import SlotRepresentation
