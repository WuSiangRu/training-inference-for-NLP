#%%
file = r"resources\gpt2\test.history_belief_action_sys_delex_with_none"

with open(file, encoding="UTF-8") as f:
    data = f.readlines()

replace_bs = False
replace_action = True
#%%
clean_lines = []
for line in data:
    history = line.split("<|belief|>")[0]
    b = line.split("<|endofcontext|>")[-1].split("<|action|>")[0]
    if replace_bs:
        b = b.replace("none", "未提及")
    if "action" in line:
        actions = line.split("<|action|>")[-1].split("<|endofaction|>")[0].split(",")
        a = list({"".join(ac.split()) for ac in actions})
        a = "<|action|>" + ",".join(a) + "<|endofaction|>"
        response = line.split("<|endofaction|>")[-1]

        clean = history + b + a + response
    else:
        clean = history + b
    clean_lines.append(clean)


#%%
if "action" in line and replace_bs:
    file_name = file + "_replace_none_repeat_action.txt"
else:
    file_name = file + "_repeat_action.txt"


with open(file_name, "wt", encoding="UTF-8") as f:
    f.writelines(clean_lines)
# %%
