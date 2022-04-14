#%%
from transformers import GPT2LMHeadModel, BertTokenizerFast


tokenizer = BertTokenizerFast.from_pretrained("ckiplab/gpt2-base-chinese")

model = GPT2LMHeadModel.from_pretrained("ckiplab/gpt2-base-chinese")

# tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#%%
# special_tokens_dict = {
#     "additional_special_tokens": [
#         "<|context|>",
#         "<|endofcontext|>",
#         "<|belief|>",
#         "<|endofbelief|>",
#         "<|response|>",
#         "<|endofresponse|>",
#         "<|dbsearch|>",
#         "<|endofdbsearch|>",
#         "<|action|>",
#         "<|endofaction|>",
#         "<|system|>",
#         "<|user|>",
#     ]
# }

print("vocab size before :", len(tokenizer))
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# print("We have added", num_added_toks, "tokens")
print("vocab size after :", len(tokenizer))

# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
model.resize_token_embeddings(len(tokenizer))
#%%
model.save_pretrained("gpt2zhtw_specialadded")
tokenizer.save_pretrained("gpt2zhtw_specialadded")

#%%
aa = tokenizer.tokenize(
    "[CLS] <|context|> <|user|> 我正在尋找價格範圍便宜的住宿地，應該是一家旅館 <|endofcontext|> <|belief|> 旅館 名稱 未提及 , 旅館 區域 未提及 , 旅館 停車處 未提及 , 旅館 價格範圍 便宜的 , 旅館 星級 未提及 , 旅館 網際網路 未提及 , 旅館 型別 酒店 <|endofbelief|> [SEP]"
)

# %%
print(aa)