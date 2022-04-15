#%%
import json
import re

with open(r"belief_states.json", encoding="UTF-8") as f:
    domain_slot_value = json.load(f)

domains = [domain for domain in domain_slot_value]
slots = {slot for domain in domain_slot_value for slot in domain_slot_value[domain]}

with open(
#    r"C:\\Users\\EB210\\Desktop\\simpletod-master\\db_checker.json", encoding="UTF-8"
     r"../simpletod-master/db_checker.json", encoding="UTF-8"
) as f:
    db_ckecker = json.load(f)

VALUE_CORRECT = {
    "芬迪頓必勝客": "必勝客",
    "必勝客迪頓": "必勝客",
    "必勝客櫻桃辛頓": "必勝客",
    "市中心必勝客": "必勝客",
    "現代歐洲的": "歐洲的",
}
# %%


def clean_belief(pred_belief):
    clean_pred_belief = []
    for bs in pred_belief:
        if any(ignore in bs for ignore in ["未提及", "不在意", "non##e", "none"]):
            continue
        clean_pred_belief.append(bs)
    return clean_pred_belief


def find_domain(pred):
    for d in ["餐廳", "旅館", "景點", "列車", "計程車"]:
        if pred.find(d) != -1:
            start, end = re.search(d, pred).span()
    if start is not None:
        pred_domain = pred[start:end]
    else:
        return None

    if pred_domain not in domains:
        return None
    return pred_domain


def convert_belief(pred_belief):
    error_msg = ""
    bs_dict_pred = {}
    start = None
    for pred in pred_belief:
        pred_domain = find_domain(pred)
        if pred_domain == "計程車":
            for s in slots:
                if pred.find(s) != -1:
                    start, end = re.search(s, pred).span()

            pred_slot = pred[start:end]
            pred_val = "".join(pred[end:].strip().lower().split())
            if pred_val in VALUE_CORRECT:
                pred_val = VALUE_CORRECT[pred_val]

            if pred_domain not in bs_dict_pred:
                bs_dict_pred[pred_domain] = {}
            bs_dict_pred[pred_domain][pred_slot] = pred_val

        else:

            for s in slots:
                if pred.find(s) != -1:
                    start, end = re.search(s, pred).span()

            pred_slot = pred[start:end]
            if pred_slot in ["名稱", "出發地", "目的地"]:
                pred_val = "".join(pred[end:].strip().lower().split())
                try:
                    if pred_val not in db_ckecker[pred_domain][pred_slot]:
                        error_msg = f"未能找到{pred_val}，請確認{pred_val}是{pred_domain}。"

                        continue
                    else:
                        pred_val = "".join(pred[end:].strip().lower().split())
                        if pred_val in VALUE_CORRECT:
                            pred_val = VALUE_CORRECT[pred_val]

                        if pred_domain not in bs_dict_pred:
                            bs_dict_pred[pred_domain] = {}
                        bs_dict_pred[pred_domain][pred_slot] = pred_val
                except:
                    error_msg = f"未能找到{pred_val}，請確認{pred_val}是{pred_domain}。"
                    continue
            else:
                pred_val = "".join(pred[end:].strip().lower().split())
                if pred_val in VALUE_CORRECT:
                    pred_val = VALUE_CORRECT[pred_val]

                if pred_domain not in bs_dict_pred:
                    bs_dict_pred[pred_domain] = {}
                bs_dict_pred[pred_domain][pred_slot] = pred_val
    return bs_dict_pred, error_msg


def get_response_new(pred_text):
    if "<|response|>" in pred_text:
        return (
            (pred_text.split("<|response|>")[-1])
            .replace("<|endofresponse|>", "")
            .replace("[SEP]", "")
        )
    else:
        return ""


#%%
def get_belief(sent):
    if "<|belief|>" in sent:
        tmp = sent.strip(" ").split("<|belief|>")[-1].split("<|action|>")[0]
    else:
        return []
    # tmp = tmp.strip(' .,')
    tmp = tmp.strip(" , ")
    tmp = tmp.strip(" ,")
    tmp = tmp.replace("<|endofbelief|>", "")
    # tmp = tmp.replace("<|endoftext|>", "")
    tmp = tmp.replace("[CLS]", "")
    tmp = tmp.replace("[SEP]", "")
    belief = tmp.split(",")
    new_belief = []
    for bs in belief:
        # bs = bs.strip(" .,")
        bs = "".join(bs.split())
        if bs not in new_belief:
            new_belief.append(bs)
    return new_belief


def get_belief_dbsearch(sent):

    if "<|belief|>" in sent:
        tmp = sent.strip(" ").split("<|belief|>")[-1].split("<|endofbelief|>")[0]
    else:
        return []
    # tmp = tmp.strip(' .,')
    tmp = tmp.strip(" , ")
    tmp = tmp.strip(" ,")
    tmp = tmp.replace("<|endofbelief|>", "")
    # tmp = tmp.replace("<|endoftext|>", "")
    tmp = tmp.replace("[CLS]", "")
    tmp = tmp.replace("[SEP]", "")
    belief = tmp.split(",")
    new_belief = []
    for bs in belief:
        # bs = bs.strip(" .,")
        bs = bs.strip()
        if bs not in new_belief:
            new_belief.append(bs)
    return new_belief


def get_action(sent):
    if "<|action|>" not in sent:
        return []
    elif "<|belief|>" in sent:
        tmp = (
            sent.split("<|belief|>")[-1]
            .split("<|response|>")[0]
            .split("<|action|>")[-1]
            .strip()
        )
    elif "<|action|>" in sent:
        tmp = sent.split("<|response|>")[0].split("<|action|>")[-1].strip()
    else:
        return []
    # tmp = tmp.strip(' .,')
    tmp = tmp.strip(" , ")
    tmp = tmp.strip(" ,")
    tmp = tmp.replace("<|endofaction|>", "")
    # tmp = tmp.replace("<|endoftext|>", "")
    tmp = tmp.replace("[CLS]", "")
    tmp = tmp.replace("[SEP]", "")
    action = tmp.split(",")
    new_action = []
    for act in action:
        act = "".join(act.split())
        if act == "":
            continue
        if act not in new_action:
            new_action.append(act)
    return new_action


def get_response(sent, tokenizer):
    if "<|response|>" in sent:
        tmp = (
            sent.split("<|belief|>")[-1]
            .split("<|action|>")[-1]
            .split("<|response|>")[-1]
        )
    else:
        return ""
    # tmp = tmp.strip(' .,')
    tmp = tmp.strip(" , ")
    tmp = tmp.strip(" ,")
    tmp = tmp.replace("<|endofresponse|>", "")
    # tmp = tmp.replace("<|endoftext|>", "")
    tmp = tmp.replace("[CLS]", "")
    tmp = tmp.replace("[SEP]", "")
    tmp = "".join(tmp.split())
    tokens = tokenizer.encode(tmp, add_special_tokens=False)
    new_tokens = []
    for tok in tokens:
        if tok in tokenizer.encode(tokenizer.sep_token):
            continue
        new_tokens.append(tok)
    # response = tokenizer.decode(new_tokens).strip(" ,.")
    response = "".join(tokenizer.decode(new_tokens).split())
    return response


def get_belief_openaigpt(sent):
    if "< | belief | >" in sent:
        tmp = sent.strip(" ").split("< | belief | >")[-1].split("< | action | >")[0]
    else:
        return []
    tmp = tmp.strip(" .,")
    tmp = tmp.replace("< | endofbelief | >", "")
    tmp = tmp.replace("< | endoftext | >", "")
    belief = tmp.split(",")
    new_belief = []
    for bs in belief:
        bs = bs.strip(" .,")
        if bs not in new_belief:
            new_belief.append(bs)
    return new_belief


def get_response_openaigpt(sent, tokenizer):
    if "< | response | >" in sent:
        tmp = (
            sent.split("< | belief | >")[-1]
            .split("< | action | >")[-1]
            .split("< | response | >")[-1]
        )
    else:
        return ""
    tmp = tmp.strip(" .,")
    tmp = tmp.replace("< | endofresponse | >", "")
    tmp = tmp.replace("< | endoftext | >", "")
    tokens = tokenizer.encode(tmp)
    new_tokens = []
    for tok in tokens:
        if tok in tokenizer.encode(tokenizer._eos_token):
            continue
        new_tokens.append(tok)
    response = tokenizer.decode(new_tokens).strip(" ,.")
    response = response.replace("[ ", "[")
    response = response.replace(" ]", "]")
    response = response.replace(" _ ", "_")
    response = response.replace("i d", "id")
    return response


def get_action_openaigpt(sent):
    if "< | belief | >" in sent:
        tmp = (
            sent.split("< | belief | >")[-1]
            .split("< | response | >")[0]
            .split("< | action | >")[-1]
            .strip()
        )
    elif "< | action | >" in sent:
        tmp = sent.split("< | response | >")[0].split("< | action | >")[-1].strip()
    else:
        return []
    tmp = tmp.strip(" .,")
    tmp = tmp.replace("< | endofaction | >", "")
    tmp = tmp.replace("< | endoftext | >", "")
    action = tmp.split(",")
    new_action = []
    for act in action:
        if act == "":
            continue
        act = act.strip(" .,")
        if act not in new_action:
            act = act.replace("i d", "id")
            new_action.append(act)
    return new_action


def get_db_dynamically(predicted_text, goal, multiwoz_db):
    # gen_belief = ["domain slot value", "domain slot value", ...]
    gen_belief = get_belief_dbsearch(predicted_text)
    belief_domain = {}
    belief_book_domain = {}
    for bs in gen_belief:
        if bs in ["", " "]:
            continue
        bs = "".join(bs.split())
        for domain in domains:
            if bs.find(domain) != -1:
                start, end = re.search(domain, bs).span()
        bs_domain = bs[start:end]
        # if "book" in bs:
        if "預定" in bs:  # eg. bs = "旅館 預定停留天數 3"
            # bs_slot = bs.split()[2]
            for slot in slots:
                if bs.find(slot) != -1:
                    start, end = re.search(slot, bs).span()
            bs_slot = bs[start:end]
            bs_val = bs[end:]
            if bs_domain not in belief_book_domain:
                belief_book_domain[bs_domain] = {}
            belief_book_domain[bs_domain][bs_slot] = bs_val
        else:
            for slot in slots:
                if bs.find(slot) != -1:
                    start, end = re.search(slot, bs).span()
            bs_slot = bs[start:end]
            bs_val = bs[end:]
            # bs_slot = bs.split()[1]
            # bs_val = " ".join(bs.split()[2:])
            if bs_domain not in belief_domain:
                belief_domain[bs_domain] = {}
                belief_book_domain[bs_domain] = {}
            belief_domain[bs_domain][bs_slot] = bs_val

    db_text_tmp = []
    fail_slots = {"stay": "預定停留天數", "day": "預定日期", "time": "預定時間"}
    time_values = {
        "monday": "星期一",
        "tuesday": "星期二",
        "wednesday": "星期三",
        "thursday": "星期四",
        "friday": "星期五",
        "saturday": "星期六",
        "sunday": "星期日",
    }
    for dom in belief_domain:
        # if dom not in ["restaurant", "hotel", "attraction", "train"]:
        if dom not in ["餐廳", "旅館", "景點", "列車"]:
            continue

        # .queryResultVenues(domain, slot_value: dict, real_belief=True)

        domain_match = len(
            multiwoz_db.queryResultVenues(dom, belief_domain[dom], real_belief=True)
        )

        # if dom != "train":
        if dom != "列車":
            domain_match_text = (
                ">=5" if domain_match >= 5 else "={}".format(domain_match)
            )
        else:
            if domain_match == 0:
                domain_match_text = "=0"
            elif domain_match == 10:
                domain_match_text = "<11"
            elif domain_match == 2:
                domain_match_text = "<3"
            elif domain_match == 40:
                domain_match_text = "<41"
            elif domain_match == 5:
                domain_match_text = "<6"
            else:
                domain_match_text = ">40"

        domains_map = {
            "餐廳": "restaurant",
            "旅館": "hotel",
            "景點": "attraction",
            "列車": "train",
            "計程車": "taxi",
            "醫院": "hospital",
            "警察機關": "police",
        }

        if "fail_book" in goal[domains_map[dom]]:
            # for item in goal[dom]["fail_book"].items():
            for slot, value in goal[domains_map[dom]]["fail_book"].items():
                slot = fail_slots[slot]
                if slot == "預定日期":
                    value = time_values[value]
                item = (slot, value)
                if item in belief_book_domain[dom].items():
                    domain_book_text = "not available"
                    break
                else:
                    domain_book_text = "available"
        # else:
        if domain_match == 0:
            domain_book_text = "not available"
        else:
            domain_book_text = "available"

        # db_text_tmp.append(
        #     "{} match{} booking={}".format(dom, domain_match_text, domain_book_text)
        # )
        # print("domain_book_text:", domain_book_text)
        db_text_tmp.append(
            "{} 符合{} 預定={}".format(dom, domain_match_text, domain_book_text)
        )

    db_text = " <|dbsearch|> {} <|endofdbsearch|>".format(" , ".join(db_text_tmp))
    return db_text
