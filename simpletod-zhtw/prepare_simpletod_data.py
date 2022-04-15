from utils.args_parser import ArgsParser
from data.dataset.multiwoz import MultiWozDataset
import en_core_web_sm
from nltk import ngrams
from utils.multiwoz import dbPointer
from pprint import pprint
import ipdb
import json
import random
import os
import sys

from transformers import BertTokenizerFast

gpt2_tokenizer = BertTokenizerFast.from_pretrained("dict/")
# gpt2_tokenizer.bos_token = gpt2_tokenizer.cls_token
# gpt2_tokenizer.eos_token = gpt2_tokenizer.sep_token


multiwoz_data = json.load(open("resources/multi-woz/lex.json", "r", encoding="UTF-8"))
save_dir = "./resources/gpt2"
os.makedirs(save_dir, exist_ok=True)

for split in ["train", "val", "test"]:

    opt = ArgsParser().parse()
    opt.use_knowledge = True
    opt.use_action = True
    opt.context_knowledge = True
    opt.lexical = True

    data = MultiWozDataset(opt, split=split, shuffle=False)

    opt_delex = ArgsParser().parse()
    data_delex = MultiWozDataset(opt_delex, split=split, shuffle=False)

    history_raw_new = []
    belief_raw_new = []
    belief_raw_none_new = []
    action_raw_new = []
    output_raw_new = []
    output_raw_delex_new = []
    db_search_raw = []
    db_nmatch_raw = []

    # if split == 'test':
    #     test_dict = {}

    lex_dict = {d["name"]: d for d in data}
    delex_dict = {d["name"]: d for d in data_delex}

    for key, d_lex in lex_dict.items():
        # key: dialogue name
        d_delex = delex_dict[key]
        inp = d_lex["input_raw"]  # all user turns' raw text
        out = d_lex["target_raw"]  # all sys turns' raw text
        out_delex = d_delex["target_raw"]
        db_data = d_lex["db"]
        goal = multiwoz_data[key]["goal"]

        for i, (usr, sys) in enumerate(zip(inp, out)):
            if i == 0:
                history_new = "<|context|> <|user|> {} <|endofcontext|>".format(usr)
            else:
                tmp_new = ["<|context|>"]
                for k in range(i):

                    tmp_new.append("<|user|> " + inp[k])
                    tmp_new.append("<|system|> " + out[k])

                tmp_new.append("<|user|> " + usr + "<|endofcontext|>")
                history_new = " ".join(tmp_new)

            # # Clean translation error in test
            # if key == "SNG01608":
            #     history_new = history_new.replace("劍橋中有葡萄牙餐廳嗎", "劍橋有葡萄牙餐廳嗎")
            # if key == "MUL2305":
            #     history_new = history_new.replace(
            #         "是的，我正在尋找一個有趣的旅遊景點，能否將我引向某些地方的退房方向", "我正在尋找一個娛樂場所，能否給我一些方向"
            #     )
            # if key == "MUL1060":
            #     history_new = history_new.replace("酒店", "飯店")
            # if key == "MUL0088":
            #     history_new = history_new.replace(
            #         "我正在尋找一家便宜的酒店，靠近劍橋，設有免費停車場", "我正在尋找一家設有免費停車場的便宜飯店，靠近劍橋"
            #     )
            # if key == "MUL1350":
            #     history_new = history_new.replace(
            #         "我確實需要在東部找到一個便宜的地方", "我需要在東部找一個便宜的地方住宿"
            #     )
            # if key == "PMUL4134":
            #     history_new = history_new.replace("我聽到有關皇后學院的一些好訊息", "我聽說皇后學院不錯")
            # if key == "MUL1139":
            #     history_new = history_new.replace(
            #         "你能幫我找到旅館嗎？我正在尋找可免費停車且價格昂貴的汽車。", "你能幫我找到可自由停車且價格昂貴的飯店嗎？"
            #     )
            # if key == "SNG02153":
            #     history_new = history_new.replace(
            #         "我需要打車去攝政畫廊接我，然後帶我到唐帕斯誇裡披薩店", "我需要打車從攝政畫廊到唐帕斯誇裡披薩店"
            #     )
            # if key == "PMUL2215":
            #     history_new = history_new.replace("讓我們在市中心尋找娛樂場所", "我想尋找市中心的娛樂場所")
            # if key == "SNG01270":
            #     history_new = history_new.replace(
            #         "我需要在伊恩大廈乘計程車於14:45之前離開", "我需要在14:45乘計程車於蘭香樓出發"
            #     )
            #     history_new = history_new.replace("伊恩大廈", "蘭香樓")
            # if key == "MUL0896":
            #     history_new = history_new.replace("我需要去倫敦的火車", "我需要買一張倫敦的火車票")
            # if key == "MUL1898":
            #     history_new = history_new.replace(
            #         "我正在尋找麥格達倫學院附近的資訊", "我正在尋找有關麥格達倫這個學院類型的學院資訊"
            #     )
            # if key == "MUL0912":
            #     history_new = history_new.replace(
            #         "我很期待嘗試當地的餐館，但希望能幫助您找到一個進城的地方。我希望它在南部和一個游泳池",
            #         "我很期待嘗試當地的餐館，但希望能在城裡找個景點去。我希望景點在南部和一個游泳池",
            #     )
            # if key == "PMUL0095":
            #     history_new = history_new.replace("europeon", "歐洲")
            # if key == "SNG0822":
            #     history_new = history_new.replace("鎮上", "市中心")
            # if key == "MUL0818":
            #     history_new = history_new.replace("不太貴", "便宜")
            # if key == "PMUL4239":
            #     history_new = history_new.replace("關於基督學院，您能告訴我什麼？", "我想知道關於基督學院的資訊。")
            # if key == "SNG0840":
            #     history_new = history_new.replace("購物中心", "百貨公司")
            #     history_new = history_new.replace("酒店", "飯店")
            # if key == "SNG02205":
            #     history_new = history_new.replace(
            #         "您可以在22:45之後幫助我到達劍橋滋意餐廳嗎", "您可以在22:45之後開車載我到達劍橋滋意餐廳嗎"
            #     )
            # if key == "PMUL1424":
            #     history_new = history_new.replace("4:45", "16:45")
            # if key == "MUL2053":
            #     history_new = history_new.replace("您能幫我找到一家2星級酒店或旅館嗎", "您能幫我找到一家2星旅店嗎")
            # if key == "PMUL2708":
            #     history_new = history_new.replace("酒店", "飯店")
            # if key == "PMUL1593":
            #     history_new = history_new.replace(
            #         "我正在尋找一列星期三出發去斯多福特主教的火車", "我正在尋找一列星期三從劍橋出發去斯多福特主教的火車"
            #     )
            # if key == "SNG1086":
            #     history_new = history_new.replace("kettle'syard", "茶壺院")
            # if key == "PMUL1788":
            #     history_new = history_new.replace(
            #         "flinchesbedandbreakfast", "芬奇住宿加早餐旅館"
            #     )
            # if key == "PMUL4011":
            #     history_new = history_new.replace(
            #         "我正在尋找住宿的地方。該酒店應該在東部，不需要包括網際網路。", "我正在尋找在東部的酒店。不在意是否包括網際網路。"
            #     )
            # if key == "MUL0739":
            #     history_new = history_new.replace("酒店", "飯店")
            # if key == "MUL2630":
            #     history_new = history_new.replace("戲劇", "飯店")
            # if key == "PMUL0090":
            #     history_new = history_new.replace(
            #         "我正在寫一篇關於劍橋沒有星星的地方的文章。您能幫我找到一個有免費wifi上網的地方嗎",
            #         "我正在寫一篇關於劍橋這個的地方的文章。您能幫我找到一個有免費wifi上網的0星地方嗎",
            #     )
            # if key == "PMUL0090":
            #     history_new = history_new.replace("酒店", "飯店")
            # if key == "PMUL2457":
            #     history_new = history_new.replace("櫻桃提示中必勝客的地址是什麼？", "必勝客櫻桃辛頓的地址是什麼？")
            # if key == "PMUL1779":
            #     history_new = history_new.replace(
            #         "我需要你找到旅館，所以我有一個住處。它不需要包括網際網路，但是應該包括免費停車場。",
            #         "我需要一間酒店。我不在意網際網路，但是應該包括免費停車場。",
            #     )
            # if key == "PMUL4294":
            #     history_new = history_new.replace("請給我一個昂貴的地方吃飯", "請給我一個位於市中心昂貴的地方吃飯")
            # if key == "PMUL1253":
            #     history_new = history_new.replace("10:00", "10:15")
            # if key == "MUL0844":
            #     history_new = history_new.replace("您能幫我找到一個基於建築的景點嗎", "您能幫我找到一個建築的景點嗎")
            # if key == "MUL0228":
            #     history_new = history_new.replace("我不想在市中心吃飯太貴或便宜", "我想在市中心吃飯，價格適中")
            # if key == "MUL2254":
            #     history_new = history_new.replace("我需要一家沒有免費停車場的旅館", "我需要一家旅館，沒有停車場的")
            # if key == "PMUL3437":
            #     history_new = history_new.replace("酒店", "飯店")
            # if key == "PMUL4344":
            #     history_new = history_new.replace(
            #         "我正在尋找一個睡覺的地方，既不是太貴，也不是便宜的地下室", "我正在尋找一個睡覺的地方，價格適中的地下室"
            #     )
            # if key == "SNG1004":
            #     history_new = history_new.replace("酒店", "飯店")
            # if key == "MUL0798":
            #     history_new = history_new.replace("酒店", "飯店")
            # if key == "PMUL1435":
            #     history_new = history_new.replace("18:00", "18:15")
            # if key == "MUL0199":
            #     history_new = history_new.replace("酒店", "飯店")
            # if key == "MUL2151":
            #     history_new = history_new.replace(
            #         "您能給我有關去劍橋的火車的資訊嗎", "您能給我有關星期三去劍橋的火車的資訊嗎"
            #     )
            # if key == "MUL1028":
            #     history_new = history_new.replace("酒店", "飯店")
            # if key == "MUL2089":
            #     history_new = history_new.replace("酒店", "飯店")
            # if key == "MUL2206":
            #     history_new = history_new.replace("酒店", "飯店")
            # if key == "PMUL3301":
            #     history_new = history_new.replace("酒店", "飯店")
            # if key == "PMUL3127":
            #     history_new = history_new.replace("星期日有開往劍橋的火車嗎", "星期日有從劍橋出發的火車嗎")
            # if key == "SNG02207":
            #     history_new = history_new.replace("我需要訂一個不同於甘地的稅", "我需要從甘地餐廳搭車")
            # if key == "PMUL3886":
            #     history_new = history_new.replace(
            #         "我正在尋找價格範圍高的不錯的餐廳，並預定4張桌子", "我正在尋找價格範圍高的不錯的餐廳，並預定4人桌"
            #     )
            # if key == "PMUL3158":
            #     history_new = history_new.replace(
            #         "我想在鎮上找到一個稱為耶穌綠色戶外游泳池的地方", "我想在鎮上找一個游泳池，名稱叫做耶穌綠色戶外游泳池"
            #     )

            sys_delex = out_delex[i]
            sys_delex = (
                sys_delex.replace("[計程車_型別]", "[taxi_type]")
                .replace("[計程車_電話]", "[taxi_phone]")
                .replace("[餐廳_名稱]", "[restaurant_name]")
            )
            history_raw_new.append(history_new)
            output_raw_new.append("<|response|> " + sys + " <|endofresponse|>")

            output_raw_delex_new.append(
                "<|response|> " + sys_delex.strip() + " <|endofresponse|>"
            )

            db_text = dbPointer.convert_dbpointer_to_text(
                db_data[i], goal, d_lex["belief_raw"][i]
            )
            db_search_raw.append("<|dbsearch|> {} <|endofdbsearch|>".format(db_text))

            db_text_nmatch = dbPointer.convert_dbpointer_to_text_nmatch(
                db_data[i], goal, d_lex["belief_raw"][i]
            )
            db_nmatch_raw.append(
                "<|dbsearch|> {} <|endofdbsearch|>".format(db_text_nmatch)
            )

        # belief_raw_new: belief for End-to-End task (exclude none)
        belief = d_lex["belief_raw"]
        for bs in belief:
            tmp_bs_new = []  # 保存 belief_raw 裡不為 'not mentioned' 的資料
            for i, b in enumerate(bs):
                # if b[-1] in ["not mentioned"]:  # comment this for DST task
                if b[-1] in ["未提及"]:
                    continue
                tmp_bs_new.append(" ".join(b))
            if not tmp_bs_new:
                tmp_bs_new.append(" ")

            tmp_new = "<|belief|> {} <|endofbelief|>".format(" , ".join(tmp_bs_new))
            belief_raw_new.append(tmp_new)

        # belief_raw_none_new: belief for DST task (include none)
        for bs in belief:
            tmp_bs_new = [" ".join(b) for i, b in enumerate(bs)]
            if not tmp_bs_new:
                tmp_bs_new.append(" ")

            tmp_new = "<|belief|> {} <|endofbelief|>".format(" , ".join(tmp_bs_new))
            # '<|belief|> hotel name not mentioned ,
            #  hotel area not mentioned ,
            #  hotel parking not mentioned ,
            #  hotel pricerange cheap ,
            #  hotel stars not mentioned ,
            #  hotel internet not mentioned ,
            #  hotel type hotel <|endofbelief|>'
            belief_raw_none_new.append(tmp_new)

        action = d_lex["action_raw"]
        for act in action:
            tmp_act_new = [" ".join(a) for i, a in enumerate(act)]
            if not tmp_act_new:
                tmp_act_new.append(" ")

            tmp_new = "<|action|> {} <|endofaction|>".format(" , ".join(tmp_act_new))
            action_raw_new.append(tmp_new)

    ####################### history_belief_dbsearch_action_sys_delex #######################
    tmp = [
        # " ".join([inp.lower(), bs.lower(), dbsearch.lower(), act, trg])
        " ".join([inp, bs, dbsearch.lower(), act, trg])
        for inp, bs, dbsearch, act, trg in zip(
            history_raw_new,
            belief_raw_new,
            db_search_raw,
            action_raw_new,
            output_raw_delex_new,
        )
    ]

    with open(
        "{}/{}.history_belief_dbsearch_action_sys_delex".format(save_dir, split),
        "wt",
        encoding="UTF-8",
    ) as f:
        for l in tmp:
            # f.write("{} {}\n".format(gpt2_tokenizer._bos_token, l.lower()))
            f.write(
                "{} {} {}\n".format(
                    gpt2_tokenizer.cls_token, l.lower(), gpt2_tokenizer.sep_token
                )
            )

    ####################### history_belief_dbnmatch_action_sys_delex #######################
    tmp = []
    for inp, bs, dbsearch, act, trg in zip(
        history_raw_new,
        belief_raw_new,
        db_nmatch_raw,
        action_raw_new,
        output_raw_delex_new,
    ):
        # tmp.append(" ".join([inp.lower(), bs.lower(), dbsearch.lower(), act, trg]))
        tmp.append(" ".join([inp, bs, dbsearch.lower(), act, trg]))
    with open(
        "{}/{}.history_belief_dbnmatch_action_sys_delex".format(save_dir, split),
        "wt",
        encoding="UTF-8",
    ) as f:
        for l in tmp:
            # f.write("{} {}\n".format(gpt2_tokenizer._bos_token, l.lower()))
            f.write(
                "{} {} {}\n".format(
                    gpt2_tokenizer.cls_token, l.lower(), gpt2_tokenizer.sep_token
                )
            )

    ####################### history #######################
    with open("{}/{}.history".format(save_dir, split), "wt", encoding="UTF-8") as f:
        for l in history_raw_new:
            # f.write("{} {}\n".format(gpt2_tokenizer._bos_token, l.lower()))
            f.write(
                "{} {} {}\n".format(
                    gpt2_tokenizer.cls_token, l.lower(), gpt2_tokenizer.sep_token
                )
            )

    ####################### history_belief #######################
    tmp = []
    for hist, bs in zip(history_raw_new, belief_raw_none_new):
        # tmp.append(" ".join([hist.lower(), bs.lower()]))
        tmp.append(" ".join([hist, bs]))
    with open(
        "{}/{}.history_belief".format(save_dir, split), "wt", encoding="UTF-8"
    ) as f:
        for l in tmp:
            # f.write(
            #     "{} {} {}\n".format(
            #         gpt2_tokenizer._bos_token, l.lower(), gpt2_tokenizer._eos_token
            #     )
            # )
            f.write(
                "{} {} {}\n".format(
                    gpt2_tokenizer.cls_token, l.lower(), gpt2_tokenizer.sep_token
                )
            )

    ####################### history_belief_w/o none #######################
    tmp = []
    for hist, bs in zip(history_raw_new, belief_raw_new):
        # tmp.append(" ".join([hist.lower(), bs.lower()]))
        tmp.append(" ".join([hist, bs]))
    with open(
        "{}/{}.history_belief_without_none".format(save_dir, split),
        "wt",
        encoding="UTF-8",
    ) as f:
        for l in tmp:
            # f.write(
            #     "{} {} {}\n".format(
            #         gpt2_tokenizer._bos_token, l.lower(), gpt2_tokenizer._eos_token
            #     )
            # )
            f.write(
                "{} {} {}\n".format(
                    gpt2_tokenizer.cls_token, l.lower(), gpt2_tokenizer.sep_token
                )
            )

    ####################### history_belief_action_sys_delex_with_none #######################
    tmp = []
    for hist, bs, act, trg in zip(
        history_raw_new, belief_raw_none_new, action_raw_new, output_raw_delex_new
    ):
        # tmp.append(" ".join([hist.lower(), bs.lower(), act, trg]))
        tmp.append(" ".join([hist, bs, act, trg]))
    with open(
        "{}/{}.history_belief_action_sys_delex_with_none".format(save_dir, split),
        "wt",
        encoding="UTF-8",
    ) as f:
        for l in tmp:
            # f.write(
            #     "{} {} {}\n".format(
            #         gpt2_tokenizer._bos_token, l.lower(), gpt2_tokenizer._eos_token
            #     )
            # )
            f.write(
                "{} {} {}\n".format(
                    gpt2_tokenizer.cls_token, l.lower(), gpt2_tokenizer.sep_token
                )
            )

    ####################### history_belief_action_sys_delex #######################
    tmp = []
    for hist, bs, act, trg in zip(
        history_raw_new, belief_raw_new, action_raw_new, output_raw_delex_new
    ):
        # tmp.append(" ".join([hist.lower(), bs.lower(), act, trg]))
        tmp.append(" ".join([hist, bs, act, trg]))
    with open(
        "{}/{}.history_belief_action_sys_delex".format(save_dir, split),
        "wt",
        encoding="UTF-8",
    ) as f:
        for l in tmp:
            # f.write(
            #     "{} {} {}\n".format(
            #         gpt2_tokenizer._bos_token, l.lower(), gpt2_tokenizer._eos_token
            #     )
            # )
            f.write(
                "{} {} {}\n".format(
                    gpt2_tokenizer.cls_token, l.lower(), gpt2_tokenizer.sep_token
                )
            )
