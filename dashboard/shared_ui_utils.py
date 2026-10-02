import os
import sys
import torch
import streamlit as st

# Add project root to path safely
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.siamese_cnn import SiameseChangeDetector
from models.transformer_cd import TransformerChangeDetector

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model(checkpoint_path):
    """
    Shared model loader.
    Returns (model, in_channels, ckpt_dict).
    """
    import config
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    name = ckpt['model_name']
    in_ch = ckpt.get('in_channels', config.IN_CHANNELS)
    if name == 'siamese':
        m = SiameseChangeDetector(in_channels=in_ch)
    else:
        m = TransformerChangeDetector(in_channels=in_ch)
    m.load_state_dict(ckpt['model_state'])
    m.eval()
    return m.to(DEVICE), in_ch, ckpt
