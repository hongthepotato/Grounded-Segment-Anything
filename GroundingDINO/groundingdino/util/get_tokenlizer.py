import logging
from pathlib import Path
from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

logger = logging.getLogger(__name__)

# Supported BERT-compatible models for multilingual/Chinese support
SUPPORTED_BERT_MODELS = {
    "bert-base-uncased",
    "bert-base-cased",
    "bert-base-multilingual-cased",
    "bert-base-multilingual-uncased",
    "bert-base-chinese",
}


def get_tokenlizer(text_encoder_type, bert_base_uncased_path):
    if not isinstance(text_encoder_type, str):
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    
    # Use local path if provided (for offline environments or custom models)
    if is_bert_model_use_local_path(bert_base_uncased_path):
        model_name = _get_model_name_from_path(bert_base_uncased_path)
        logger.info("Loading tokenizer from local path: %s", bert_base_uncased_path)
        logger.info("  Model variant: %s", model_name)
        return AutoTokenizer.from_pretrained(bert_base_uncased_path)

    logger.info("Loading tokenizer from HuggingFace: %s", text_encoder_type)
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    return tokenizer


def get_pretrained_language_model(text_encoder_type, bert_base_uncased_path):
    # Use local path if provided (for offline environments or custom models)
    if is_bert_model_use_local_path(bert_base_uncased_path):
        model_name = _get_model_name_from_path(bert_base_uncased_path)
        logger.info("Loading BERT model from local path: %s", bert_base_uncased_path)
        logger.info("  Model variant: %s", model_name)
        return BertModel.from_pretrained(bert_base_uncased_path)

    # BERT-family models
    if text_encoder_type in SUPPORTED_BERT_MODELS:
        logger.info("Loading BERT model from HuggingFace: %s", text_encoder_type)
        return BertModel.from_pretrained(text_encoder_type)

    # RoBERTa models
    if text_encoder_type == "roberta-base":
        logger.info("Loading RoBERTa model from HuggingFace: %s", text_encoder_type)
        return RobertaModel.from_pretrained(text_encoder_type)

    raise ValueError(
        "Unknown text_encoder_type: {}\n"
        "Supported BERT models: {}\n"
        "Supported RoBERTa models: roberta-base".format(
            text_encoder_type, SUPPORTED_BERT_MODELS
        )
    )


def _get_model_name_from_path(bert_path: str) -> str:
    """Extract model name from local path for logging."""
    path = Path(bert_path)
    # The folder name is typically the model name (e.g., "bert-base-multilingual-cased")
    return path.name

def is_bert_model_use_local_path(bert_base_uncased_path):
    return bert_base_uncased_path is not None and len(bert_base_uncased_path) > 0
