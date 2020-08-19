from transformers import BertTokenizer
from tokenization_kobert import KoBertTokenizer


class Preprocessor:
    def __init__(self, model_type, max_len):
        self.max_len = max_len

        if model_type == "bert-base-multilingual-cased":
            self.tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=False)
        elif model_type == "monologg/kobert":
            self.tokenizer = KoBertTokenizer.from_pretrained(model_type)
        elif model_type == "etri/korbert":
            pass
        
    def transform(self, text):
        return "[CLS] {} [SEP]".format(text)

    def get_input_id(self, text):
        text = self.transform(text)
        tokenized_text = self.tokenizer.tokenize(text)

        if len(tokenized_text) > self.max_len:
            tokenized_text = tokenized_text[:self.max_len-2]

        input_id = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        input_id = input_id + ([self.tokenizer.pad_token_id] * (self.max_len-len(input_id)))
        
        return input_id

    def get_attention_mask(self, input_id):
        return [float(i > 0) for i in input_id] 