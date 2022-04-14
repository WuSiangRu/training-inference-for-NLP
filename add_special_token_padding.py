#%%
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained("gpt2")

tokenizer.add_special_tokens({"pad_token": "[PAD]"})
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
model.save_pretrained("gpt2baseline")
tokenizer.save_pretrained("gpt2baseline")

#%%
tokenizer.tokenize(
    "<|endoftext|> <|context|> <|user|> am looking for a place to to stay that has cheap price range it should be in a type of hotel <|endofcontext|> <|belief|> hotel name not mentioned , hotel area not mentioned , hotel parking not mentioned , hotel pricerange cheap , hotel stars not mentioned , hotel internet not mentioned , hotel type hotel <|endofbelief|> <|endoftext|>"
)
