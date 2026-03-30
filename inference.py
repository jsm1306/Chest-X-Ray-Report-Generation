"""
Inference Module - Generate medical reports from images using Vision-Language Model
"""
import logging
import numpy as np
import cv2
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

logger = logging.getLogger(__name__)


def generate_report(
    img_path: str,
    encoder_model,
    full_model,
    tokenizer,
    max_len: int = 200,
    mean: float = 0.485,
    std: float = 0.229
) -> str:
    """
    Generate medical report from chest X-ray image using trained Vision-Language Model.
    
    IMPORTANT: This function matches the exact architecture used in train.py:
    - Encoder outputs: (1, 49, 512) - NOT flattened
    - Full model expects: [encoder_features (1,49,512), decoder_input (1,max_len-1)]
    - Inference uses greedy decoding with the full_model
    
    Args:
        img_path: Path to input medical image
        encoder_model: Compiled encoder model (feature extractor)
        full_model: Compiled full model (encoder+decoder for inference)
        tokenizer: Keras tokenizer fitted on medical reports vocabulary
        max_len: Maximum sequence length (must match training, default 200)
        mean: Normalization mean (default 0.485)
        std: Normalization std dev (default 0.229)
        
    Returns:
        Generated medical report text
        
    Raises:
        ValueError: If image not found or invalid
        Exception: If model inference fails
    """
    try:
        # ============ 1) READ AND PREPROCESS IMAGE ============
        logger.info(f"Loading image from {img_path}")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Image not found or cannot be read: {img_path}")
        
        # Resize to model's expected input size
        img = cv2.resize(img, (224, 224))
        
        # Normalize: scale to [0,1], then standardize
        img = img / 255.0
        img = (img - mean) / std
        
        # Add batch and channel dimensions: (224, 224) -> (1, 224, 224, 1)
        img = np.expand_dims(img, axis=[0, -1])
        logger.info(f"Image preprocessed: shape {img.shape}")
        
        # ============ 2) EXTRACT ENCODER FEATURES ============
        # CRITICAL: Encoder output is (batch, 49, 512) - NOT (batch, 512)
        # This is passed directly to full_model - do NOT flatten or reshape!
        logger.info("Extracting visual features from encoder...")
        enc_feats = encoder_model.predict(img, verbose=0)
        logger.info(f"Features extracted: shape {enc_feats.shape}")
        
        if enc_feats.shape != (1, 49, 512):
            logger.warning(f"Unexpected encoder output shape: {enc_feats.shape}. Expected (1, 49, 512)")
        
        # ============ 3) GET TOKENIZER SPECIAL TOKENS ============
        # Find start token (try multiple naming conventions)
        start_token = (
            tokenizer.word_index.get('<start>') or
            tokenizer.word_index.get('startseq') or
            tokenizer.word_index.get('sos') or
            tokenizer.word_index.get('start') or
            1  # fallback to common convention
        )
        
        # Find end token (try multiple naming conventions)
        end_token = (
            tokenizer.word_index.get('<end>') or
            tokenizer.word_index.get('endseq') or
            tokenizer.word_index.get('eos') or
            tokenizer.word_index.get('end') or
            2  # fallback to common convention
        )
        
        logger.info(f"Token IDs - Start: {start_token}, End: {end_token}")
        
        # ============ 4) GENERATE REPORT TEXT (GREEDY DECODING) ============
        logger.info(f"Generating report (max_len={max_len})...")
        
        decoder_maxlen = max_len - 1  # decoder input length (199 for max_len=200)
        input_seq = [start_token]  # Start with start token
        result_words = []
        
        for step in range(decoder_maxlen):
            # Pad input sequence to fixed length used during training
            dec_inp = pad_sequences([input_seq], maxlen=decoder_maxlen, padding='post')
            
            # Get predictions for all positions
            # full_model expects: [encoder_features (1,49,512), decoder_input (1,decoder_maxlen)]
            # Returns logits shape: (1, decoder_maxlen, vocab_size)
            preds = full_model.predict([enc_feats, dec_inp], verbose=0)
            
            # Get logits for the CURRENT position to predict (next token)
            pos = len(input_seq) - 1
            
            if pos >= preds.shape[1]:
                logger.warning("Position index exceeded model output length")
                break
            
            probs = preds[0, pos]  # Probabilities for next token (vocab_size,)
            
            # Greedy selection: pick token with highest probability
            next_id = int(np.argmax(probs))
            
            # Stop if end token reached
            if next_id == end_token:
                logger.info(f"End token generated at step {step}")
                break
            
            # Skip padding tokens and start tokens
            if next_id == 0 or next_id == start_token:
                input_seq.append(next_id)
                continue
            
            # Convert token ID to word
            word = tokenizer.index_word.get(next_id, "")
            
            if word:
                result_words.append(word)
            
            input_seq.append(next_id)
        
        # ============ 5) POST-PROCESS OUTPUT ============
        if not result_words:
            logger.warning("No words generated, returning default response")
            return "Findings:\nUnable to generate report. Please try again with a different image."
        
        # Join words
        report = " ".join(result_words)
        
        # Remove token markers (only <unk>, keep legitimate text)
        report = re.sub(r"<unk>", " ", report)
        report = re.sub(r"<start>|<end>|startseq|endseq", " ", report)
        
        # Clean up whitespace
        report = re.sub(r"\s+", " ", report).strip()
        
        # Ensure proper ending punctuation
        if report and not report.endswith(('.', '!', '?')):
            report += "."
        
        # Capitalize first letter
        if report:
            report = report[0].upper() + report[1:]
        
        logger.info(f"Report generated successfully ({len(report)} chars)")
        return "Findings:\n" + report
        
        final_report = "Findings:\n" + report
        logger.info("Report generation completed successfully")
        
        return final_report
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise
