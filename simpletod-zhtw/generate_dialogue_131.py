import torch
from transformers import BertTokenizerFast, GPT2LMHeadModel
from utils.args_parser import ArgsParser
from data.dataset.multiwoz import MultiWozDataset
from evaluate_multiwoz import MultiWozDB
from utils.multiwoz import dbPointer
from utils.simpletod import *
from tqdm import tqdm
import json
import sys


opt = ArgsParser().parse()
opt.multiwoz_version = "2.1"
opt.use_action = True
opt.use_knowledge = True
opt.context_knowledge = True
opt.lexical = True


HISTORY_LEN = None
# USE_ORACLE_BELIEF = True
USE_ORACLE_BELIEF = opt.use_oracle_belief
# USE_ORACLE_ACTION = False
USE_ORACLE_ACTION = opt.use_oracle_action
# USE_DB_SEARCH = True
USE_DB_SEARCH = opt.use_db_search
USE_DYNAMIC_DB = opt.use_dynamic_db
EVAL_SPLIT = "test"
# EVAL_SPLIT = opt.split_set

decoding = opt.decoding

multiwoz_data = json.load(open("resources/multi-woz/lex.json", "r", encoding="UTF-8"))


multiwoz_db = MultiWozDB()

opt_delex = ArgsParser().parse()
opt_delex.multiwoz_version = "2.1"


data = MultiWozDataset(opt, split=EVAL_SPLIT, shuffle=False)

data_delex = MultiWozDataset(opt_delex, split=EVAL_SPLIT, shuffle=False)

lex_dict = {}
delex_dict = {}
for d in data:
    lex_dict[d["name"]] = d

for d in data_delex:
    delex_dict[d["name"]] = d

model_checkpoint = r"delex_end2end_with_none_repeat_action_size5_deepspeed"
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
model.eval()
model.to("cuda")

break_tokens = tokenizer.encode(
    tokenizer.sep_token, add_special_tokens=False
)  # "[SEP]": 102
MAX_LEN = model.config.n_ctx  # 1024


generated_dict = {}
num_data = len(data)

for i, dial_name in enumerate(tqdm(lex_dict)):
    if EVAL_SPLIT == "train" and i > 1000:
        # if EVAL_SPLIT == "train" or i > 3:
        break
    d = lex_dict[dial_name]
    d_delex = delex_dict[dial_name]
    # print("{} [{}/{}] \r".format(d["name"], i, num_data), end="")
    sys.stdout.flush()
    beliefs_raw = d["belief_raw"]
    user = d["input_raw"]
    system = d["target_raw"]
    system_delex = d_delex["target_raw"]
    if "delex" in model_checkpoint:
        target_response = system_delex
    else:
        target_response = system

    action = d["action_raw"]
    target_action = []
    for turn_act in action:
        turn_action = []
        for act in turn_act:
            act_str = "{} {} {}".format(act[0], act[1], act[2])
            act_str = "".join(act_str.split())
            turn_action.append(act_str)  # act_str: "計程車通知電話" "domain action slot"
        target_action.append(turn_action)

    # save target belief
    dialogue_aggregated_target_belief = []
    dialogue_target_belief = []
    for turn_belief in beliefs_raw:
        turn_belief_str = []
        for bs in turn_belief:
            domain, slot, value = bs
            # if value in ["not mentioned", "none"]:
            # 忽略 value:未提及, none
            if value in ["未提及", "none"]:
                continue
            # bs_str = "{} {} {}".format(domain.lower(), slot.lower(), value.lower())
            bs_str = "{} {} {}".format(domain, slot, value.lower())
            bs_str = "".join(bs_str.split())
            if bs_str not in dialogue_aggregated_target_belief:
                dialogue_aggregated_target_belief.append(bs_str)
            turn_belief_str.append(bs_str)
        dialogue_target_belief.append(sorted(turn_belief_str))

    db_data = d["db"]
    goal = multiwoz_data[dial_name]["goal"]

    generated = []
    model_context = []
    for turn_id, (usr_turn, _) in enumerate(zip(user, system)):

        if turn_id == 0:
            tmp_text = "<|user|> {}".format(usr_turn.strip())
        else:
            tmp = []
            for k in range(turn_id):
                tmp.append("<|user|> {}".format(user[k].strip()))
                tmp.append("<|system|> {}".format(system[k].strip()))

            tmp.append("<|user|> {}".format(usr_turn.strip()))

            # trim history
            if HISTORY_LEN and len(tmp) > HISTORY_LEN:
                tmp = tmp[-1 * HISTORY_LEN :]
            tmp_text = " ".join(tmp)

        if dial_name == "SNG02319":
            # tmp_text = tmp_text.replace("300 will", "03:00 will")
            tmp_text = tmp_text.replace("300", "03:00")

        text = "{} <|context|> {} <|endofcontext|> ".format(
            tokenizer.cls_token, tmp_text
        )

        # clean translation error
        if dial_name == "SNG01608":
            text = text.replace("劍橋中有葡萄牙餐廳嗎", "劍橋有葡萄牙餐廳嗎")
        if dial_name == "MUL2305":
            text = text.replace(
                "是的，我正在尋找一個有趣的旅遊景點，能否將我引向某些地方的退房方向", "我正在尋找一個娛樂場所，能否給我一些方向"
            )
        if dial_name == "MUL1060":
            text = text.replace("酒店", "飯店")
        if dial_name == "MUL0088":
            text = text.replace("我正在尋找一家便宜的酒店，靠近劍橋，設有免費停車場", "我正在尋找一家設有免費停車場的便宜飯店，靠近劍橋")
        if dial_name == "MUL1350":
            text = text.replace("我確實需要在東部找到一個便宜的地方", "我需要在東部找一個便宜的地方住宿")
        if dial_name == "PMUL4134":
            text = text.replace("我聽到有關皇后學院的一些好訊息", "我聽說皇后學院不錯")
        if dial_name == "MUL1139":
            text = text.replace(
                "你能幫我找到旅館嗎？我正在尋找可免費停車且價格昂貴的汽車。", "你能幫我找到可自由停車且價格昂貴的飯店嗎？"
            )
        if dial_name == "SNG02153":
            text = text.replace("我需要打車去攝政畫廊接我，然後帶我到唐帕斯誇裡披薩店", "我需要打車從攝政畫廊到唐帕斯誇裡披薩店")
        if dial_name == "PMUL2215":
            text = text.replace("讓我們在市中心尋找娛樂場所", "我想尋找市中心的娛樂場所")
        if dial_name == "SNG01270":
            text = text.replace("我需要在伊恩大廈乘計程車於14:45之前離開", "我需要在14:45乘計程車於蘭香樓出發")
            text = text.replace("伊恩大廈", "蘭香樓")
        if dial_name == "MUL0896":
            text = text.replace("我需要去倫敦的火車", "我需要買一張倫敦的火車票")
        if dial_name == "MUL1898":
            text = text.replace("我正在尋找麥格達倫學院附近的資訊", "我正在尋找有關麥格達倫這個學院類型的學院資訊")
        if dial_name == "MUL0912":
            text = text.replace(
                "我很期待嘗試當地的餐館，但希望能幫助您找到一個進城的地方。我希望它在南部和一個游泳池",
                "我很期待嘗試當地的餐館，但希望能在城裡找個景點去。我希望景點在南部和一個游泳池",
            )
        if dial_name == "PMUL0095":
            text = text.replace("europeon", "歐洲")
        if dial_name == "SNG0822":
            text = text.replace("鎮上", "市中心")
        if dial_name == "MUL0818":
            text = text.replace("不太貴", "便宜")
        if dial_name == "PMUL4239":
            text = text.replace("關於基督學院，您能告訴我什麼？", "我想知道關於基督學院的資訊。")
        if dial_name == "SNG0840":
            text = text.replace("購物中心", "百貨公司")
            text = text.replace("酒店", "飯店")
        if dial_name == "SNG02205":
            text = text.replace("您可以在22:45之後幫助我到達劍橋滋意餐廳嗎", "您可以在22:45之後開車載我到達劍橋滋意餐廳嗎")
        if dial_name == "PMUL1424":
            text = text.replace("4:45", "16:45")
        if dial_name == "MUL2053":
            text = text.replace("您能幫我找到一家2星級酒店或旅館嗎", "您能幫我找到一家2星旅店嗎")
        if dial_name == "PMUL2708":
            text = text.replace("酒店", "飯店")
        if dial_name == "PMUL1593":
            text = text.replace("我正在尋找一列星期三出發去斯多福特主教的火車", "我正在尋找一列星期三從劍橋出發去斯多福特主教的火車")
        if dial_name == "SNG1086":
            text = text.replace("kettle'syard", "茶壺院")
        if dial_name == "PMUL1788":
            text = text.replace("flinchesbedandbreakfast", "芬奇住宿加早餐旅館")
        if dial_name == "PMUL4011":
            text = text.replace(
                "我正在尋找住宿的地方。該酒店應該在東部，不需要包括網際網路。", "我正在尋找在東部的酒店。不在意是否包括網際網路。"
            )
        if dial_name == "MUL0739":
            text = text.replace("酒店", "飯店")
        if dial_name == "MUL2630":
            text = text.replace("戲劇", "飯店")
        if dial_name == "PMUL0090":
            text = text.replace(
                "我正在寫一篇關於劍橋沒有星星的地方的文章。您能幫我找到一個有免費wifi上網的地方嗎",
                "我正在寫一篇關於劍橋這個的地方的文章。您能幫我找到一個有免費wifi上網的0星地方嗎",
            )
        if dial_name == "PMUL0090":
            text = text.replace("酒店", "飯店")
        if dial_name == "PMUL2457":
            text = text.replace("櫻桃提示中必勝客的地址是什麼？", "必勝客櫻桃辛頓的地址是什麼？")
        if dial_name == "PMUL1779":
            text = text.replace(
                "我需要你找到旅館，所以我有一個住處。它不需要包括網際網路，但是應該包括免費停車場。",
                "我需要一間酒店。我不在意網際網路，但是應該包括免費停車場。",
            )
        if dial_name == "PMUL4294":
            text = text.replace("請給我一個昂貴的地方吃飯", "請給我一個位於市中心昂貴的地方吃飯")
        if dial_name == "PMUL1253":
            text = text.replace("10:00", "10:15")
        if dial_name == "MUL0844":
            text = text.replace("您能幫我找到一個基於建築的景點嗎", "您能幫我找到一個建築的景點嗎")
        if dial_name == "MUL0228":
            text = text.replace("我不想在市中心吃飯太貴或便宜", "我想在市中心吃飯，價格適中")
        if dial_name == "MUL2254":
            text = text.replace("我需要一家沒有免費停車場的旅館", "我需要一家旅館，沒有停車場的")
        if dial_name == "PMUL3437":
            text = text.replace("酒店", "飯店")
        if dial_name == "PMUL4344":
            text = text.replace("我正在尋找一個睡覺的地方，既不是太貴，也不是便宜的地下室", "我正在尋找一個睡覺的地方，價格適中的地下室")
        if dial_name == "SNG1004":
            text = text.replace("酒店", "飯店")
        if dial_name == "MUL0798":
            text = text.replace("酒店", "飯店")
        if dial_name == "PMUL1435":
            text = text.replace("18:00", "18:15")
        if dial_name == "MUL0199":
            text = text.replace("酒店", "飯店")
        if dial_name == "MUL2151":
            text = text.replace("您能給我有關去劍橋的火車的資訊嗎", "您能給我有關星期三去劍橋的火車的資訊嗎")
        if dial_name == "MUL1028":
            text = text.replace("酒店", "飯店")
        if dial_name == "MUL2089":
            text = text.replace("酒店", "飯店")
        if dial_name == "MUL2206":
            text = text.replace("酒店", "飯店")
        if dial_name == "PMUL3301":
            text = text.replace("酒店", "飯店")
        if dial_name == "PMUL3127":
            text = text.replace("星期日有開往劍橋的火車嗎", "星期日有從劍橋出發的火車嗎")
        if dial_name == "SNG02207":
            text = text.replace("我需要訂一個不同於甘地的稅", "我需要從甘地餐廳搭車")
        if dial_name == "PMUL3886":
            text = text.replace("我正在尋找價格範圍高的不錯的餐廳，並預定4張桌子", "我正在尋找價格範圍高的不錯的餐廳，並預定4人桌")
        if dial_name == "PMUL3158":
            text = text.replace("我想在鎮上找到一個稱為耶穌綠色戶外游泳池的地方", "我想在鎮上找一個游泳池，名稱叫做耶穌綠色戶外游泳池")

        # ORACLE: Target
        if USE_ORACLE_BELIEF:
            turn_belief = dialogue_target_belief[turn_id]
            belief_str = "<|belief|> {} <|endofbelief|>".format(" , ".join(turn_belief))
            text = text + " " + belief_str

        # The best result ignores DB Search results entirely.
        db_text = dbPointer.convert_dbpointer_to_text(
            db_data[turn_id], goal, beliefs_raw[turn_id]
        )

        if USE_DB_SEARCH and USE_ORACLE_BELIEF:
            if not USE_ORACLE_BELIEF:
                # if use oracle db, oracle belief should also be used.
                print("warning: oracle db is true, oracle belief is false")
            text += " <|dbsearch|> {} <|endofdbsearch|>".format(db_text)

        if USE_ORACLE_ACTION:
            turn_action = target_action[turn_id]
            action_str = "<|action|> {} <|endofaction|>".format(" , ".join(turn_action))
            text = text + " " + action_str

        # text: [CLS]<|context|> <|user|> <|system|> ... <|user|> <|endofcontext|>
        # <|belief|> ... <|endofbelief|>         ====oracle==== if USE_ORACLE_BELIEF = True
        # <|dbsearch|> db_text <|endofdbsearch|> ====oracle==== if USE_DB_SEARCH = True
        # <|action|> ... <|endofaction|>         ====oracle==== if USE_ORACLE_ACTION = True
        model_context.append(text)
        indexed_tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(indexed_tokens) > MAX_LEN:
            indexed_tokens = indexed_tokens[-1 * MAX_LEN :]

        # Convert indexed tokens in a PyTorch tensor, 2D
        tokens_tensor = torch.tensor([indexed_tokens])

        # If you have a GPU, put everything on cuda
        tokens_tensor = tokens_tensor.to("cuda")
        predicted_index = indexed_tokens[-1]

        if (
            USE_DB_SEARCH and not USE_ORACLE_BELIEF
        ):  # generate belief, then get DB search results, then continue generation (greedy decoding)
            with torch.no_grad():
                while predicted_index not in break_tokens:
                    outputs = model(tokens_tensor)
                    predictions = outputs[0]
                    predicted_index = torch.argmax(predictions[0, -1, :]).item()
                    indexed_tokens += [predicted_index]
                    tokens_tensor = torch.tensor([indexed_tokens]).to("cuda")
                    if len(indexed_tokens) > MAX_LEN:
                        break
                    if tokenizer.decode(indexed_tokens).endswith("<|endofbelief|>"):
                        break

            tmp_pred = tokenizer.decode(indexed_tokens)

            if not USE_DYNAMIC_DB:  # use oracle db,
                # text: [CLS]<|context|> <|user|> <|system|> ... <|user|> <|endofcontext|>
                # <|belief|> ... <|endofbelief|>         ====generated==== if USE_ORACLE_BELIEF = False
                # <|dbsearch|> db_text <|endofdbsearch|> =====oracle=====  if USE_DB_SEARCH = True, USE_DYNAMIC_DB = False
                text = "{} {}".format(tmp_pred, db_text)

            else:  # compute db search dynamically using generated belief
                db_text_dynamic = get_db_dynamically(
                    tmp_pred, goal, multiwoz_db=multiwoz_db
                )
                # text: [CLS]<|context|> <|user|> <|system|> ... <|user|> <|endofcontext|>
                # <|belief|> ... <|endofbelief|> ====generated====
                # <|dbsearch|> db_text <|endofdbsearch|> ====dynamic====
                text = "{} {}".format(tmp_pred, db_text_dynamic)

            # continue generation
            indexed_tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(indexed_tokens) > MAX_LEN:
                indexed_tokens = indexed_tokens[-1 * MAX_LEN :]

            # Convert indexed tokens in a PyTorch tensor
            tokens_tensor = torch.tensor([indexed_tokens])

            # If you have a GPU, put everything on cuda
            tokens_tensor = tokens_tensor.to("cuda")
            predicted_index = indexed_tokens[-1]

            # Predict all tokens
            with torch.no_grad():
                # while predicted_index not in break_tokens:
                while predicted_index not in break_tokens:
                    outputs = model(tokens_tensor)
                    predictions = outputs[0]
                    predicted_index = torch.argmax(predictions[0, -1, :]).item()
                    indexed_tokens += [predicted_index]

                    # sometime model generate repeated actions, we just use truncate actions if this happens
                    predicted_text = tokenizer.decode(indexed_tokens)
                    if "<|action|>" in predicted_text:
                        generated_actions = (
                            predicted_text.split("<|action|>")[-1]
                            .split("<|endofaction|>")[0]
                            .split(",")
                        )
                        new_actions = []
                        for a in generated_actions:
                            if a in ["", " "]:
                                continue
                            new_actions.append(a.strip())
                        len_actions = len(new_actions)
                        if len(list(set(new_actions))) > len(new_actions) or (
                            len_actions > 10 and not truncate_action
                        ):
                            actions = "<|action|> {} <|endofaction|>".format(
                                " , ".join(list(set(new_actions)))
                            )
                            indexed_tokens = tokenizer.encode(
                                "{} {}".format(
                                    predicted_text.split("<|action|>")[0],
                                    actions,
                                    add_special_tokens=False,
                                )
                            )
                            truncate_action = True

                    tokens_tensor = torch.tensor([indexed_tokens]).to("cuda")
                    if len(indexed_tokens) > MAX_LEN:
                        break
                    if tokenizer.decode(indexed_tokens).endswith("<|endofresponse|>"):
                        break

                predicted_text = tokenizer.decode(indexed_tokens)
                generated.append(predicted_text)

        else:  # generate belief, action, and response once (ignore DB Search results entirely)
            # text: [CLS]<|context|> <|user|> <|system|> ... <|user|> <|endofcontext|>
            # USE_ORACLE_BELIEF = False
            # USE_ORACLE_ACTION = False
            text += "<|belief|>"
            indexed_tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(indexed_tokens) > MAX_LEN:
                indexed_tokens = indexed_tokens[-1 * MAX_LEN :]

            # # Convert indexed tokens in a PyTorch tensor, 2D
            tokens_tensor = torch.tensor([indexed_tokens])

            # # If you have a GPU, put everything on cuda
            tokens_tensor = tokens_tensor.to("cuda")
            predicted_index = indexed_tokens[-1]

            with torch.no_grad():

                if decoding == "nucleus":
                    sample_output = model.generate(
                        tokens_tensor,
                        do_sample=True,
                        max_length=MAX_LEN,
                        top_p=0.5,
                        top_k=0,
                    )
                    predicted_text = tokenizer.decode(sample_output[0])
                    tmp = " ".join(
                        [
                            predicted_text.split("<|endofresponse|>")[0],
                            "<|endofresponse|>",
                        ]
                    )
                    predicted_text = tmp
                    generated.append(predicted_text)

                elif decoding == "greedy":
                    truncate_action = False
                    while predicted_index not in break_tokens:
                        outputs = model(tokens_tensor)
                        predictions = outputs[0]
                        predicted_index = torch.argmax(predictions[0, -1, :]).item()
                        indexed_tokens += [predicted_index]

                        # sometime model generate repeated actions, we just use truncate actions if this happens
                        predicted_text = tokenizer.decode(indexed_tokens)
                        # if "<|action|>" in predicted_text:
                        #     generated_actions = (
                        #         predicted_text.split("<|action|>")[-1]
                        #         .split("<|endofaction|>")[0]
                        #         .split(",")
                        #     )
                        #     new_actions = []
                        #     for a in generated_actions:
                        #         if a in ["", " "]:
                        #             continue
                        #         new_actions.append(a.strip())
                        #     len_actions = len(new_actions)
                        #     if len(list(set(new_actions))) > len(new_actions) or (
                        #         len_actions > 10 and not truncate_action
                        #     ):
                        #         actions = "<|action|> {} <|endofaction|>".format(
                        #             " , ".join(list(set(new_actions)))
                        #         )
                        #         indexed_tokens = tokenizer.encode(
                        #             "{} {}".format(
                        #                 predicted_text.split("<|action|>")[0],
                        #                 actions,
                        #                 add_special_tokens=False,
                        #             )
                        #         )
                        #         truncate_action = True

                        # Modified
                        if "<|endofaction|>" in predicted_text and not truncate_action:
                            generated_actions = (
                                predicted_text.split("<|action|>")[-1]
                                .split("<|endofaction|>")[0]
                                .split(",")
                            )
                            new_actions = []
                            for a in generated_actions:
                                if a in ["", " "]:
                                    continue
                                new_actions.append("".join(a.split()))
                            actions = list(set(new_actions))
                            actions = ",".join(actions)
                            actions = f"<|action|> {actions} <|endofaction|>"
                            predicted_text = (
                                predicted_text.split("<|action|>")[0] + actions
                            )
                            indexed_tokens = tokenizer.encode(
                                predicted_text, add_special_tokens=False
                            )
                            truncate_action = True
                        # EndOfModified

                        tokens_tensor = torch.tensor([indexed_tokens]).to("cuda")
                        if len(indexed_tokens) > MAX_LEN:
                            break
                        decode_text = "".join(tokenizer.decode(indexed_tokens).split())
                        if (
                            "<|response|>" in decode_text
                            and "<|endofresponse|>" in decode_text
                        ):
                            break

                    predicted_text = tokenizer.decode(indexed_tokens)
                   # if "<|endofresponse|>" not in predicted_text:
                   #     predicted_text += "<|endofresponse|>"

                    predicted_text = "".join(predicted_text.split())
                    generated.append(predicted_text)

    dialogue_aggregated_pred_belief = []
    dialogue_pred_belief = []
    dialogue_pred_responses = []
    dialogue_pred_action = []

    # aggregate belief states
    for turn, pred in enumerate(generated):
        turn_pred_belief = []
        if "openai-gpt" in model_checkpoint:
            belief = get_belief_openaigpt(pred)
        else:
            if (
                "dbsearch" in model_checkpoint
                or "dbnmatch" in model_checkpoint
                or USE_DB_SEARCH
                or "db" in model_checkpoint
            ):
                belief = get_belief_dbsearch(pred)
            else:
                belief = get_belief(pred)
        if len(belief) > 0:
            for bs in belief:
                if bs not in ["", " "] and bs not in dialogue_aggregated_pred_belief:
                    dialogue_aggregated_pred_belief.append(bs)
            new_belief = list(set(belief))
            dialogue_pred_belief.append(sorted(new_belief))
        else:
            if len(dialogue_pred_belief) == 0:
                dialogue_pred_belief.append([""])
            else:
                dialogue_pred_belief.append(dialogue_pred_belief[-1])
        if "openai-gpt" in model_checkpoint:
            gen_response = get_response_openaigpt(pred, tokenizer)
        else:
            gen_response = get_response(pred, tokenizer)
        dialogue_pred_responses.append(gen_response)

        if "openai-gpt" in model_checkpoint:
            gen_action = get_action_openaigpt(pred)
        else:
            gen_action = get_action(pred)
        dialogue_pred_action.append(gen_action)

    generated_dict[d["name"]] = {
        "target_belief": dialogue_aggregated_target_belief,
        "target_turn_belief": dialogue_target_belief,
        "generated_belief": dialogue_aggregated_pred_belief,
        "generated_turn_belief": dialogue_pred_belief,
        "target_response": target_response,
        "generated_response": dialogue_pred_responses,
        "target_action": target_action,
        "generated_action": dialogue_pred_action,
        "target_user": user,
        "model_context": model_context,
    }


save_name = "{}_{}".format(model_checkpoint, EVAL_SPLIT)

if USE_ORACLE_BELIEF:
    save_name += "_oracleBelief"


if USE_DB_SEARCH:
    save_name += "_oracleDB"

if USE_DYNAMIC_DB:
    save_name += "_dynamicDB"

if USE_ORACLE_ACTION:
    save_name += "_oracleAction"


if HISTORY_LEN:
    save_name += "_context[history={}]".format(HISTORY_LEN)
else:
    save_name += "_context[history=full_history]"

# save_name += "_nocarry"
#save_name += "test_clean"
save_name += "test_clean_epoch3"
with open("{}.json".format(save_name), "wt", encoding="UTF-8") as f:
    json.dump(generated_dict, f, ensure_ascii=False, indent=4)
