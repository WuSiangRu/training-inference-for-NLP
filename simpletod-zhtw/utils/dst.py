import json
import re


with open(
   # r"C:\Users\EB210\Desktop\simpletod-master\db_checker.json", encoding="UTF-8"
     r"../simpletod-master/db_checker.json", encoding="UTF-8"
) as f:
    db_ckecker = json.load(f)

# GENERAL_TYPO = {
#     # type
#     "guesthouse": "guest house",
#     "guesthouses": "guest house",
#     "guest": "guest house",
#     "mutiple sports": "multiple sports",
#     "sports": "multiple sports",
#     "mutliple sports": "multiple sports",
#     "swimmingpool": "swimming pool",
#     "concerthall": "concert hall",
#     "concert": "concert hall",
#     "pool": "swimming pool",
#     "night club": "nightclub",
#     "mus": "museum",
#     "ol": "architecture",
#     "colleges": "college",
#     "coll": "college",
#     "architectural": "architecture",
#     "musuem": "museum",
#     "churches": "church",
#     # area
#     "center": "centre",
#     "center of town": "centre",
#     "near city center": "centre",
#     "in the north": "north",
#     "cen": "centre",
#     "east side": "east",
#     "east area": "east",
#     "west part of town": "west",
#     "ce": "centre",
#     "town center": "centre",
#     "centre of cambridge": "centre",
#     "city center": "centre",
#     "the south": "south",
#     "scentre": "centre",
#     "town centre": "centre",
#     "in town": "centre",
#     "north part of town": "north",
#     "centre of town": "centre",
#     "cb30aq": "none",
#     # price
#     "mode": "moderate",
#     "moderate -ly": "moderate",
#     "mo": "moderate",
#     # day
#     "next friday": "friday",
#     "monda": "monday",
#     # parking
#     "free parking": "free",
#     # internet
#     "free internet": "yes",
#     # star
#     "4 star": "4",
#     "4 stars": "4",
#     "0 star rarting": "none",
#     # others
#     "y": "yes",
#     "any": "dontcare",
#     "n": "no",
#     "does not care": "dontcare",
#     "not men": "none",
#     "not": "none",
#     "not mentioned": "none",
#     "": "none",
#     "not mendtioned": "none",
#     "3 .": "3",
#     "does not": "no",
#     "fun": "none",
#     "art": "none",
# }
VALUE_CORRECT = {
    "芬迪頓必勝客": "必勝客",
    "必勝客迪頓": "必勝客",
    "必勝客櫻桃辛頓": "必勝客",
    "市中心必勝客": "必勝客",
    "現代歐洲的": "歐洲的",
}

IGNORE_TURNS_TYPE2 = {
    "PMUL1812": [1, 2],
    "MUL2177": [9, 10, 11, 12, 13, 14],
    "PMUL0182": [1],
    "PMUL0095": [1],
    "MUL1883": [1],
    "PMUL2869": [9, 11],
    "SNG0433": [0, 1],
    "PMUL4880": [2],
    "PMUL2452": [2],
    "PMUL2882": [2],
    "SNG01391": [1],
    "MUL0803": [7],
    "MUL1560": [4, 5],
    "PMUL4964": [6, 7],
    "MUL1753": [8, 9],
    "PMUL3921": [4],
    "PMUL3403": [0, 4],
    "SNG0933": [3],
    "SNG0296": [1],
    "SNG0477": [1],
    "MUL0814": [1],
    "SNG0078": [1],
    "PMUL1036": [6],
    "PMUL4840": [2],
    "PMUL3423": [6],
    "MUL2284": [2],
    "PMUL1373": [1],
    "SNG01538": [1],
    "MUL0011": [2],
    "PMUL4326": [4],
    "MUL1697": [10],
    "MUL0014": [5],
    "PMUL1370": [1],
    "PMUL1801": [7],
    "MUL0466": [2],
    "PMUL0506": [1, 2],
    "SNG1036": [2],
}

with open(
    #r"C:\Users\EB210\Desktop\simpletod-master\utils\belief_states.json",
     r"../simpletod-master/belief_states.json",
    encoding="UTF-8",
) as f:
    domain_slot_value = json.load(f)

domains = [domain for domain in domain_slot_value]
slots = {slot for domain in domain_slot_value for slot in domain_slot_value[domain]}


def ignore_none(pred_belief, target_belief):
    # for pred in pred_belief:
    #     if 'catherine s' in pred:
    #         pred.replace('catherine s', 'catherines')

    clean_target_belief = []
    clean_pred_belief = []
    for bs in target_belief:
        # 忽略 value:未提及、不在意
        # if "not mentioned" in bs:
        if any(ignore in bs for ignore in ["未提及", "不在意", "non##e"]):
            continue
        clean_target_belief.append(bs)

    for bs in pred_belief:
        # 忽略 value:未提及、不在意
        # if "not mentioned" in bs:
        if any(ignore in bs for ignore in ["未提及", "不在意", "non##e"]):
            continue
        clean_pred_belief.append(bs)

    target_belief = clean_target_belief
    pred_belief = clean_pred_belief

    return pred_belief, target_belief


def fix_mismatch_jason(slot, value):
    # miss match slot and value
    if (
        slot == "type"
        and value
        in [
            "nigh",
            "moderate -ly priced",
            "bed and breakfast",
            "centre",
            "venetian",
            "intern",
            "a cheap -er hotel",
        ]
        or slot == "internet"
        and value == "4"
        or slot == "pricerange"
        and value == "2"
        or slot == "type"
        and value in ["gastropub", "la raza", "galleria", "gallery", "science", "m"]
        or "area" in slot
        and value in ["moderate"]
        or "day" in slot
        and value == "t"
    ):
        value = "none"
    elif slot == "type" and value in [
        "hotel with free parking and free wifi",
        "4",
        "3 star hotel",
    ]:
        value = "hotel"
    elif slot == "star" and value == "3 star hotel":
        value = "3"
    elif "area" in slot:
        if value == "no":
            value = "north"
        elif value == "we":
            value = "west"
        elif value == "cent":
            value = "centre"
    elif "day" in slot:
        if value == "we":
            value = "wednesday"
        elif value == "no":
            value = "none"
    elif "price" in slot and value == "ch":
        value = "cheap"
    elif "internet" in slot and value == "free":
        value = "yes"

    # some out-of-define classification slot values
    if (
        slot == "area"
        and value in ["stansted airport", "cambridge", "silver street"]
        or slot == "area"
        and value in ["norwich", "ely", "museum", "same area as hotel"]
    ):
        value = "none"
    return slot, value


def default_cleaning(pred_belief, target_belief, dial, turn_id, task="orginal"):
    domains = ["餐廳", "旅館", "景點", "列車", "計程車"]
    pred_belief_jason = []
    target_belief_jason = []
    bs_dict_pred = {}
    bs_dict_target = {}
    for pred in pred_belief:
        if pred in ["", " "]:
            continue

        if dial == "PMUL3647":
            pred = pred.replace("[UNK]", "")
            pred = pred.replace("和萵苣餐廳", "蛞蝓和萵苣餐廳")
        if dial == "MUL0099":
            pred = pred.replace("[UNK]", "梔")
        if dial == "PMUL3976":
            pred = pred.replace("[UNK]", "梔")
        if "餐廳食物墨西哥的" in pred and dial == "SNG0446":
            continue
        if dial == "63":
            pred = pred.replace("拉米莫薩餐廳", "lamimosa")
            pred = pred.replace("洛弗爾酒店", "洛弗爾旅館酒店")
        if dial == "327" and turn_id == 0:
            continue
        if dial == "327" and "列車" in pred and turn_id == 3:
            continue
        if dial == "296" and "停車處" in pred:
            continue
        if dial == "27" and turn_id == 0:
            continue
        if dial == "202" and "景點區域中心" in pred:
            continue
        if dial == "202" and "旅館型別" in pred:
            continue
        if dial == "959" and turn_id == 1 and "劍橋貝爾弗萊酒店" in pred:
            continue
        if dial == "959" and turn_id == 1 and "網際網路" in pred:
            continue
        if dial == "848" and "餐廳食物" in pred:
            continue
        if dial == "803" and turn_id == 0 and "酒店" in pred:
            continue
        if dial == "803" and turn_id == 4 and "出發地" in pred:
            continue
        if dial == "803" and turn_id == 4 and "預定人數" in pred:
            continue
        if dial == "390" and "旅館型別" in pred:
            continue
        if dial == "390" and "網際網路" in pred:
            continue
        if dial == "390" and turn_id == 3 and "列車" in pred:
            continue
        if dial == "201" and "國際的" in pred:
            continue
        if dial == "365" and "星級" in pred:
            continue
        if dial == "853":
            pred = pred.replace("薩代艾爾別墅酒店", "薩代艾爾")
        if dial == "853" and "網際網路" in pred:
            continue
        if dial == "522" and "景點型別" in pred:
            continue
        if dial == "617" and "中心" in pred:
            continue
        if dial == "766" and "旅館型別" in pred:
            continue
        if dial == "766" and turn_id == 4 and "阿什利酒店" in pred:
            continue
        if dial == "766" and "預定人數" in pred:
            continue
        if dial == "269" and "景點型別" in pred:
            continue
        if dial == "201" and turn_id == 1 and "普雷佐餐廳" in pred:
            continue
        if dial == "201" and turn_id == 3 and "日期" in pred:
            continue
        if dial == "201" and turn_id == 3 and "預定人數" in pred:
            continue
        if dial == "201" and turn_id == 3 and "預定時間" in pred:
            continue
        if dial == "201" and turn_id == 4 and "日期" in pred:
            continue
        if dial == "201" and turn_id == 4 and "預定人數" in pred:
            continue
        if dial == "201" and turn_id == 4 and "預定時間" in pred:
            continue
        if dial == "52" and "景點區域" in pred:
            continue
        if dial == "52" and "景點型別" in pred:
            continue
        if dial == "52" and turn_id == 3 and "餐廳名稱" in pred:
            continue
        if dial == "960" and "網際網路" in pred:
            continue
        if dial == "618" and "餐廳日期" in pred:
            continue
        if task == "dstc9":
            if "網際網路" in pred:
                pred = pred.replace("網際網路有", "網際網路是的")
            if "停車處" in pred:
                pred = pred.replace("停車處有", "停車處是的")

        if "預定日期" in pred:
            pred = pred.replace("預定日期", "日期")
        if "倫敦國王十字區站" in pred:
            pred = pred.replace("倫敦國王十字區站", "倫敦國王十字區")
        if "<|endofresponse|>" in pred:
            pred = pred.replace("<|endofresponse|>", "")

        start = None
        for d in domains:
            if pred.find(d) != -1:
                start, end = re.search(d, pred).span()
        if start is not None:
            pred_domain = pred[start:end]
        else:
            continue

        if pred_domain not in domains:
            continue

        if pred_domain == "計程車":
            for s in slots:
                if pred.find(s) != -1:
                    start, end = re.search(s, pred).span()

            pred_slot = pred[start:end]
            pred_val = "".join(pred[end:].strip().lower().split())
            if pred_val in VALUE_CORRECT:
                pred_val = VALUE_CORRECT[pred_val]

            pred_belief_jason.append(
                "{} {} {}".format(pred_domain, pred_slot, pred_val)
            )
            if pred_domain not in bs_dict_pred:
                bs_dict_pred[pred_domain] = {}
            bs_dict_pred[pred_domain][pred_slot] = pred_val

        else:
            # slot = pred.split()[1]
            # val = " ".join(pred.split()[2:])
            for s in slots:
                if pred.find(s) != -1:
                    start, end = re.search(s, pred).span()

            pred_slot = pred[start:end]
            if pred_slot in ["名稱", "出發地", "目的地"]:
                pred_val = "".join(pred[end:].strip().lower().split())
                try:
                    if pred_val not in db_ckecker[pred_domain][pred_slot]:
                        continue
                    else:
                        pred_val = "".join(pred[end:].strip().lower().split())
                        if pred_val in VALUE_CORRECT:
                            pred_val = VALUE_CORRECT[pred_val]

                        pred_belief_jason.append(
                            "{} {} {}".format(pred_domain, pred_slot, pred_val)
                        )
                        if pred_domain not in bs_dict_pred:
                            bs_dict_pred[pred_domain] = {}
                        bs_dict_pred[pred_domain][pred_slot] = pred_val
                except:
                    continue
            else:
                pred_val = "".join(pred[end:].strip().lower().split())
                if pred_val in VALUE_CORRECT:
                    pred_val = VALUE_CORRECT[pred_val]

                pred_belief_jason.append(
                    "{} {} {}".format(pred_domain, pred_slot, pred_val)
                )
                if pred_domain not in bs_dict_pred:
                    bs_dict_pred[pred_domain] = {}
                bs_dict_pred[pred_domain][pred_slot] = pred_val

    for tgt in target_belief:
        # domain = tgt.split()[0]
        if "預定日期" in tgt:
            tgt = tgt.replace("預定日期", "日期")
        if "預定停留天數" in tgt:
            tgt = tgt.replace("晚", "")

        if dial == "327" and "泥爐炭火烹飪法宮" in tgt:
            continue
        if dial == "201" and turn_id == 0 and "義大利" in tgt:
            continue
        if dial == "201" and "義大利" in tgt:
            continue
        if dial == "80":
            tgt = tgt.replace("8:30", "08:30")
        if dial == "365" and "丘吉爾學院公寓酒店" in tgt:
            tgt = tgt.replace("丘吉爾學院公寓酒店", "丘吉爾學院")
        if dial == "898" and "預定人數" in tgt:
            tgt = tgt.replace("2", "3")
        if dial == "992" and turn_id == 3 and "預定人數" in tgt:
            continue
        if dial == "617" and "小酒館" in tgt:
            continue
        if dial == "617" and "中心" in tgt:
            continue
        if dial == "819" and "泥爐炭火烹飪法宮" in tgt:
            continue
        if dial == "833" and "小酒館" in tgt:
            continue
        if dial == "766" and "旅館型別" in tgt:
            continue
        if dial == "529":
            tgt = tgt.replace("8:15", "08:15")
        if dial == "269" and "丘吉爾學院公寓酒店" in tgt:
            tgt = tgt.replace("丘吉爾學院公寓酒店", "丘吉爾學院")
        if dial == "52" and "景點型別" in tgt:
            continue
        if dial == "52" and "景點區域" in tgt:
            continue
        if dial == "960" and "網際網路" in tgt:
            continue
        if dial == "177" and "網際網路" in tgt:
            continue

        for d in domains:
            if tgt.find(d) != -1:
                start, end = re.search(d, tgt).span()

        tgt_domain = tgt[start:end]
        if tgt_domain not in domains:
            continue

        if tgt_domain == "計程車":
            for s in slots:
                if tgt.find(s) != -1:
                    start, end = re.search(s, tgt).span()

            tgt_slot = tgt[start:end]
            tgt_val = "".join(tgt[end:].strip().lower().split())
            if tgt_val in VALUE_CORRECT:
                tgt_val = VALUE_CORRECT[tgt_val]

            target_belief_jason.append("{} {} {}".format(tgt_domain, tgt_slot, tgt_val))
            if tgt_domain not in bs_dict_target:
                bs_dict_target[tgt_domain] = {}
            bs_dict_target[tgt_domain][tgt_slot] = tgt_val

        else:
            # slot = tgt.split()[1]
            # val = " ".join(tgt.split()[2:])
            for s in slots:
                if tgt.find(s) != -1:
                    start, end = re.search(s, tgt).span()

            tgt_slot = tgt[start:end]
            if tgt_slot in ["名稱", "出發地", "目的地"]:
                tgt_val = "".join(tgt[end:].strip().lower().split())
                if tgt_val not in db_ckecker[tgt_domain][tgt_slot]:
                    continue
                else:
                    tgt_val = "".join(tgt[end:].strip().lower().split())
                    if tgt_val in VALUE_CORRECT:
                        tgt_val = VALUE_CORRECT[tgt_val]

                    if tgt_slot == "到達時間" and tgt_val == "劍橋":
                        continue

                    target_belief_jason.append(
                        "{} {} {}".format(tgt_domain, tgt_slot, tgt_val)
                    )
                    if tgt_domain not in bs_dict_target:
                        bs_dict_target[tgt_domain] = {}
                    bs_dict_target[tgt_domain][tgt_slot] = tgt_val
            else:
                tgt_val = "".join(tgt[end:].strip().lower().split())
                if tgt_val in VALUE_CORRECT:
                    tgt_val = VALUE_CORRECT[tgt_val]
                target_belief_jason.append(
                    "{} {} {}".format(tgt_domain, tgt_slot, tgt_val)
                )
                if tgt_domain not in bs_dict_target:
                    bs_dict_target[tgt_domain] = {}
                bs_dict_target[tgt_domain][tgt_slot] = tgt_val

    turn_pred = pred_belief_jason
    turn_target = target_belief_jason

    return turn_pred, turn_target, bs_dict_pred, bs_dict_target
