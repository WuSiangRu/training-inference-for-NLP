#%%
import json

path = r"resources\gpt2\train.history_belief"

with open(path, encoding="UTF-8") as f:
    data = f.readlines()

# %%
domain_slot_value = {}

for line in data:
    bs = line.split("<|belief|>")[1].split("<|endofbelief|>")[0]

    for dom_slot_val in bs.split(","):
        dom_slot_val = dom_slot_val.strip()
        if dom_slot_val:
            domain = dom_slot_val.split()[0]
            slot = dom_slot_val.split()[1]
            value = "".join(dom_slot_val.split()[2:])
            if domain not in domain_slot_value:
                domain_slot_value[domain] = {}

            if slot not in domain_slot_value[domain]:
                domain_slot_value[domain][slot] = []

            if value not in domain_slot_value[domain][slot]:
                domain_slot_value[domain][slot].append(value)

# %%
with open("belief_states.json", "w", encoding="UTF-8") as f:
    # with open("belief_states_dstc9.json", "w", encoding="UTF-8") as f:
    json.dump(domain_slot_value, f, ensure_ascii=False, indent=4)

# %%
