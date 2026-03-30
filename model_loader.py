"""
Model Loader - Loads models once at startup for thread-safe inference
"""
import os
import pickle
import logging
from typing import Optional
import numpy as np
from tensorflow import keras
import tensorflow as tf

logger = logging.getLogger(__name__)

# Global model variables
encoder_model = None
full_model = None
tokenizer = None


# Custom Attention Layer - Must match training implementation exactly
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = keras.layers.Dense(units)
        self.W2 = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)

    def call(self, encoder_output, hidden_state):
        hidden_state_with_time_axis = tf.expand_dims(hidden_state, 1)
        score = tf.nn.tanh(self.W1(encoder_output) + self.W2(hidden_state_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


# Custom functions for model compatibility
def apply_attention(inputs):
    """
    Custom attention function matching training implementation.
    Applies Bahdanau attention to encoder outputs based on decoder state.
    """
    encoder_out, decoder_emb = inputs
    hidden_state = tf.reduce_mean(encoder_out, axis=1)  # Get hidden state from encoder output
    attention = BahdanauAttention(512)  # Match training units
    context_vector, attn_weights = attention(encoder_out, hidden_state)
    
    # Expand and repeat context to match decoder time steps
    context_vector = tf.expand_dims(context_vector, 1)
    context_vector = tf.repeat(context_vector, repeats=tf.shape(decoder_emb)[1], axis=1)
    
    # Concatenate context with decoder embedding
    combined = tf.concat([context_vector, decoder_emb], axis=-1)
    return combined


def load_models(
    encoder_path: str = "encoder_model.keras",
    full_model_path: str = "full_model.keras",
    tokenizer_path: str = "tokenizer.pkl"
) -> tuple:
    """
    Load all models and tokenizer at startup.
    Models are loaded only once and cached globally.
    
    Args:
        encoder_path: Path to encoder model
        full_model_path: Path to full model
        tokenizer_path: Path to tokenizer pickle file
        
    Returns:
        Tuple of (encoder_model, full_model, tokenizer)
        
    Raises:
        FileNotFoundError: If model files don't exist
        Exception: If model loading fails
    """
    global encoder_model, full_model, tokenizer
    
    try:
        # Validate file existence
        for path, name in [
            (encoder_path, "Encoder model"),
            (full_model_path, "Full model"),
            (tokenizer_path, "Tokenizer")
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found at {path}")
        
        # Define custom objects for model loading
        custom_objects = {
            'apply_attention': apply_attention,
            'BahdanauAttention': BahdanauAttention,
        }
        
        # Load encoder model
        logger.info(f"Loading encoder model from {encoder_path}...")
        encoder_model = keras.models.load_model(
            encoder_path,
            custom_objects=custom_objects,
            safe_mode=False
        )
        logger.info("✓ Encoder model loaded")
        
        # Load full model
        logger.info(f"Loading full model from {full_model_path}...")
        full_model = keras.models.load_model(
            full_model_path,
            custom_objects=custom_objects,
            safe_mode=False
        )
        logger.info("✓ Full model loaded")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}...")
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
        logger.info("✓ Tokenizer loaded")
        logger.info("✓ All models loaded successfully!")
        
        return encoder_model, full_model, tokenizer
        
    except FileNotFoundError as e:
        logger.error(f"❌ Model file error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise


def get_models() -> tuple:
    """
    Get globally loaded models.
    
    Returns:
        Tuple of (encoder_model, full_model, tokenizer)
        
    Raises:
        RuntimeError: If models haven't been loaded yet
    """
    if encoder_model is None or full_model is None or tokenizer is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")
    return encoder_model, full_model, tokenizer


def unload_models():
    """Unload models from memory"""
    global encoder_model, full_model, tokenizer
    encoder_model = None
    full_model = None
    tokenizer = None
    logger.info("Models unloaded from memory")
