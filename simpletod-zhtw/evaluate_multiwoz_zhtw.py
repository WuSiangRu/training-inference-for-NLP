import math

# import utils.delexicalize as delex
from utils.multiwoz import delexicalize as delex
from collections import Counter
from nltk.util import ngrams
import json
from utils.multiwoz.nlp import normalize, normalize_for_sql
import sqlite3
import os
import random
import sys
from utils.multiwoz.nlp import BLEUScorer
from utils.dst import ignore_none, default_cleaning
from transformers import BertTokenizerFast


def remove_model_mismatch_and_db_data(
    dial_name, target_beliefs, pred_beliefs, domain, t
):
    if (
        domain == "hotel"
        and domain in target_beliefs[t]
        and "type" in pred_beliefs[domain]
    ):
        if "type" in target_beliefs[t][domain]:
            if pred_beliefs[domain]["type"] != target_beliefs[t][domain]["type"]:
                pred_beliefs[domain]["type"] = target_beliefs[t][domain]["type"]
        else:
            del pred_beliefs[domain]["type"]

    if (
        "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "pizza hut fenditton"
    ):
        pred_beliefs[domain]["name"] = "pizza hut fen ditton"

    if (
        domain == "restaurant"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "riverside brasserie"
    ):
        pred_beliefs[domain]["food"] = "modern european"

    if (
        domain == "restaurant"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "charlie chan"
    ):
        pred_beliefs[domain]["area"] = "centre"

    if (
        domain == "restaurant"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "saint johns chop house"
    ):
        pred_beliefs[domain]["pricerange"] = "moderate"

    if (
        domain == "restaurant"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "pizza hut fen ditton"
    ):
        pred_beliefs[domain]["pricerange"] = "moderate"

    if (
        domain == "restaurant"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "cote"
    ):
        pred_beliefs[domain]["pricerange"] = "expensive"

    if (
        domain == "restaurant"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "cambridge lodge restaurant"
    ):
        pred_beliefs[domain]["food"] = "european"

    if (
        domain == "restaurant"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "cafe jello gallery"
    ):
        pred_beliefs[domain]["food"] = "peking restaurant"

    if (
        domain == "restaurant"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "nandos"
    ):
        pred_beliefs[domain]["food"] = "portuguese"

    if (
        domain == "restaurant"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "yippee noodle bar"
    ):
        pred_beliefs[domain]["pricerange"] = "moderate"

    if (
        domain == "restaurant"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "copper kettle"
    ):
        pred_beliefs[domain]["food"] = "british"

    if (
        domain == "restaurant"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] in ["nirala", "the nirala"]
    ):
        pred_beliefs[domain]["food"] = "indian"

    if (
        domain == "attraction"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "vue cinema"
    ) and "type" in pred_beliefs[domain]:
        del pred_beliefs[domain]["type"]

    if (
        domain == "attraction"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "funky fun house"
    ):
        pred_beliefs[domain]["area"] = "dontcare"

    if (
        domain == "attraction"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "little seoul"
    ):
        pred_beliefs[domain][
            "name"
        ] = "downing college"  # correct name in turn_belief_pred

    if (
        domain == "attraction"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "byard art"
    ):
        pred_beliefs[domain]["type"] = "museum"  # correct name in turn_belief_pred

    if (
        domain == "attraction"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "trinity college"
    ):
        pred_beliefs[domain]["type"] = "college"  # correct name in turn_belief_pred

    if (
        domain == "attraction"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "cambridge university botanic gardens"
    ):
        pred_beliefs[domain]["area"] = "centre"  # correct name in turn_belief_pred

    if (
        domain == "hotel"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "lovell lodge"
    ):
        pred_beliefs[domain]["parking"] = "yes"  # correct name in turn_belief_pred

    if (
        domain == "hotel"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "whale of a time"
    ):
        pred_beliefs[domain][
            "type"
        ] = "entertainment"  # correct name in turn_belief_pred

    if (
        domain == "hotel"
        and "name" in pred_beliefs[domain]
        and pred_beliefs[domain]["name"] == "a and b guest house"
    ):
        pred_beliefs[domain]["parking"] = "yes"  # correct name in turn_belief_pred

    if (
        dial_name == "MUL0116.json"
        and domain == "hotel"
        and "area" in pred_beliefs[domain]
    ):
        del pred_beliefs[domain]["area"]

    return pred_beliefs


class BaseEvaluator(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp):
        raise NotImplementedError

    def get_report(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn


class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def score(self, hypothesis, corpus, n=1):
        # containers
        tokenizer = BertTokenizerFast.from_pretrained("dict")
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in zip(hypothesis, corpus):
            if type(hyps[0]) is list:
                # hyps = [hyp.split() for hyp in hyps[0]]
                hyps = [tokenizer.tokenize(hyp) for hyp in hyps[0]]
            else:
                # hyps = [hyp.split() for hyp in hyps]
                hyps = [tokenizer.tokenize(hyp) for hyp in hyps]

            # refs = [ref.split() for ref in refs]
            refs = [tokenizer.tokenize(ref) for ref in refs]

            for idx, hyp in enumerate(hyps):
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = {
                        ng: min(count, max_counts[ng]) for ng, count in hypcnts.items()
                    }
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0:
                        break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)
                if n == 1:
                    break
        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 for i in range(4)]
        s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)
        return bp * math.exp(s)


class MultiWozDB(object):
    # loading databases
    domains = ["restaurant", "hotel", "attraction", "train", "taxi", "hospital"]
    dbs = {}
    dbs_en = {}
    CUR_DIR = os.path.dirname(__file__)

    for domain in domains:
        # db = os.path.join("utils/multiwoz/db/{}-dbase.db".format(domain))
        db = os.path.join("utils/multiwoz/db/zhtw_db/zhtw_{}_dbase.db".format(domain))
        conn = sqlite3.connect(db)
        c = conn.cursor()
        dbs[domain] = c

    for domain in domains:
        db = os.path.join("utils/multiwoz/db/{}-dbase.db".format(domain))
        # db = os.path.join("utils/multiwoz/db/zhtw_db/zhtw_{}_dbase.db".format(domain))
        conn = sqlite3.connect(db)
        c = conn.cursor()
        dbs_en[domain] = c

    def queryResultVenues(self, domain, turn, real_belief=False, en=False):
        tables_map = {
            "餐廳": "RESTAURANTS",
            "旅館": "HOTELS",
            "景點": "ATTRACTIONS",
            "列車": "TRAINS",
            "restaurant": "RESTAURANTS",
            "hotel": "HOTELS",
            "attraction": "ATTRACTIONS",
            "train": "TRAINS",
        }

        # query the db
        table = tables_map[domain]
        sql_query = "select * from {}".format(table)

        if real_belief == True:
            items = turn.items()
        else:
            domains_map = {
                "restaurant": "餐廳",
                "hotel": "旅館",
                "attraction": "景點",
                "train": "列車",
            }
            # items = turn["metadata"][domain]["semi"].items()
            items = turn["metadata"][domains_map[domain]]["semi"].items()

        ignore_vals = [
            "",
            "dontcare",
            "not mentioned",
            "don't care",
            "dont care",
            "do n't care",
            "不在意",
            "未提及",
            "none",
            "一晚",
        ]
        ignore_keys = ["預定人數", "預定時間", "預定停留天數"]
        flag = True
        for key, val in items:
            if key == "日期" and table != "TRAINS":
                continue
            if val not in ignore_vals and key not in ignore_keys:
                # if val not in ["不在意"]:
                val2 = val.replace("'", "''")
                if key not in ["arriveBy", "leaveAt", "到達時間", "出發時間"]:
                    val2 = normalize(val2)
                if flag:
                    sql_query += " where "
                    if key in ["arriveBy", "到達時間"]:
                        sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                    elif key in ["leaveAt", "出發時間"]:
                        sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    if key in ["arriveBy", "到達時間"]:
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    elif key in ["leaveAt", "出發時間"]:
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

        try:  # "select * from attraction where name = 'queens college'"
            domains_map = {
                "餐廳": "restaurant",
                "旅館": "hotel",
                "景點": "attraction",
                "列車": "train",
                "計程車": "taxi",
                "醫院": "hospital",
                "警察機關": "police",
            }
            if domain in domains_map:
                domain = domains_map[domain]
            if "a&b賓館" in sql_query:
                sql_query = sql_query.replace("a&b賓館", " A&B 賓館")

            if not en:
                # sql_query = sql_query.replace("未提及", "")
                # sql_query = sql_query.replace("不在意", "")
                # sql_query = sql_query.replace("none", "")
                fetch_result = self.dbs[domain].execute(sql_query).fetchall()
            else:
                sql_query = sql_query.replace("RESTAURANTS", "restaurant")
                sql_query = sql_query.replace("HOTELS", "hotel")
                sql_query = sql_query.replace("ATTRACTIONS", "attraction")
                sql_query = sql_query.replace("TRAINS", "train")
                fetch_result = self.dbs_en[domain].execute(sql_query).fetchall()
            # print("SQL_QUERY:", sql_query)

            return fetch_result

        except Exception as e:
            print("SQL_QUERY:", sql_query)
            print("ErrorMessage:", e)
            sys.exit()
            return []  # TODO test it


class MultiWozEvaluator(BaseEvaluator):
    def __init__(self, data_name):
        self.data_name = data_name
        self.slot_dict = delex.prepareSlotValuesIndependent()
        self.delex_dialogues = json.load(
            open("resources/multi-woz/delex.json", "r", encoding="UTF-8")
        )
        self.db = MultiWozDB()
        self.labels = []
        self.hyps = []
        # self.venues = json.load(open("resources/all_venues.json", "r"))

    def add_example(self, ref, hyp):
        self.labels.append(ref)
        self.hyps.append(hyp)

    def _parseGoal(self, goal, d, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {"informable": [], "requestable": [], "booking": []}
        if "info" in d["goal"][domain]:

            # we consider dialogues only where train had to be booked!
            # requestable
            if "book" in d["goal"][domain]:
                goal[domain]["requestable"].append("reference")

            if domain == "train":
                if (
                    "reqt" in d["goal"][domain]
                    and "trainID" in d["goal"][domain]["reqt"]
                ):
                    goal[domain]["requestable"].append("id")
            else:
                if "reqt" in d["goal"][domain]:
                    for s in d["goal"][domain]["reqt"]:  # addtional requests:
                        if s in ["phone", "address", "postcode", "reference", "id"]:
                            goal[domain]["requestable"].append(s)

            # informable
            goal[domain]["informable"] = d["goal"][domain]["info"]

            # booking
            if "book" in d["goal"][domain]:
                goal[domain]["booking"] = d["goal"][domain]["book"]

        return goal

    def _evaluateGeneratedDialogue(
        self, dialname, dial, goal, realDialogue, real_requestables, soft_acc=False
    ):
        """Evaluates the dialogue created by the model.
        First we load the user goal of the dialogue, then for each turn
        generated by the system we look for key-words.
        For the Inform rate we look whether the entity was proposed.
        Inform rate: measures how often the entities provided by the system are correct.
        Inform rate 簡單來說 就是根據生成的 belief states 條件對資料庫進行條件搜索
        如果依生成條件所搜索的結果 存在於 dialog 的 goal 所設條件所搜尋的結果內，則視為 MATCH (底下的MATCH即為Inform)

        For the Success rate we look for requestables slots
        Success rate: refers to how often the system is able to answer all the requested attributes by user.
        Success rate 簡單來說 就是 根據生成的句子結果蒐集各 domain 的 requestables 並保存至 provided_requestables
        接著將 provided_requestables 和 real_requestables 相比對
        如果生成的 requestables 結果 皆符合 real_requestables，則視為 SUCCESS
        """

        # For computing corpus success 之後用來計算success rate
        # requestable slots 為資料庫可提供的資訊，如資料庫可提供電話、地址、郵遞區號、參考編號、火車ID、各domain名稱
        requestables = ["phone", "address", "postcode", "reference", "id"]

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}  # for success rate
        venue_offered = {}  # for inform rate
        domains_in_goal = []

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        m_targetutt = [
            turn["text"] for idx, turn in enumerate(realDialogue["log"]) if idx % 2 == 1
        ]

        pred_beliefs = dial["beliefs"]
        target_beliefs = dial["target_beliefs"]
        pred_responses = dial["responses"]
        domains_map = {
            "restaurant": "餐廳",
            "hotel": "旅館",
            "attraction": "景點",
            "train": "列車",
        }

        for t, (sent_gpt, sent_t) in enumerate(zip(pred_responses, m_targetutt)):
            sent_t = (
                sent_t.replace("計程車_型別", "taxi_type")
                .replace("計程車_電話", "taxi_phone")
                .replace("餐廳_名稱", "restaurant_name")
            )
            sent_gpt = (
                sent_gpt.replace("計程車_型別", "taxi_type")
                .replace("計程車_電話", "taxi_phone")
                .replace("餐廳_名稱", "restaurant_name")
            )
            for domain in goal.keys():

                # 如果生成句含 [domain_name] 或 [domain_id]
                # if "[" + domain + "_name]" in sent_gpt or "_id" in sent_gpt:
                if "[" + domain + "_" in sent_gpt or "_id" in sent_gpt:
                    if domain in sent_gpt:
                        if domain in ["restaurant", "hotel", "attraction", "train"]:

                            if domains_map[domain] not in pred_beliefs[t]:
                                venues = []
                            else:
                                # remove_model_mismatch_and_db_data 只適用於原版英文資料集
                                # pred_beliefs = remove_model_mismatch_and_db_data(
                                #     dialname, target_beliefs, pred_beliefs[t], domain, t
                                # )
                                # print("Pred db result:")
                                venues = self.db.queryResultVenues(
                                    domains_map[domain],
                                    pred_beliefs[t][domains_map[domain]],
                                    real_belief=True,
                                )

                            venue_offered[domain] = venues
                            # # if venue has changed
                            # if len(venue_offered[domain]) == 0 and venues:
                            #     venue_offered[domain] = venues
                            # else:
                            #     flag = False
                            #     for ven in venues:
                            #         if venue_offered[domain][0] == ven:
                            #             flag = True
                            #             break
                            #     if (
                            #         not flag and venues
                            #     ):  # sometimes there are no results so sample won't work
                            #         venue_offered[domain] = venues

                        else:  # not limited so we can provide one
                            venue_offered[domain] = "[" + domain + "_name]"

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == "reference":
                        if domain + "_reference" in sent_gpt:
                            if "restaurant_reference" in sent_gpt:
                                if (
                                    realDialogue["log"][t * 2]["db_pointer"][-5] == 1
                                ):  # if pointer was allowing for that?
                                    provided_requestables[domain].append("reference")

                            elif "hotel_reference" in sent_gpt:
                                if (
                                    realDialogue["log"][t * 2]["db_pointer"][-3] == 1
                                ):  # if pointer was allowing for that?
                                    provided_requestables[domain].append("reference")

                            elif "train_reference" in sent_gpt:
                                if (
                                    realDialogue["log"][t * 2]["db_pointer"][-1] == 1
                                ):  # if pointer was allowing for that?
                                    provided_requestables[domain].append("reference")

                            else:
                                provided_requestables[domain].append("reference")
                    else:

                        if domain + "_" + requestable + "]" in sent_gpt:
                            provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            if "info" in realDialogue["goal"][domain]:
                if "name" in realDialogue["goal"][domain]["info"]:
                    venue_offered[domain] = "[" + domain + "_name]"

            # special domains that entity does not need to be provided
            if domain in ["taxi", "police", "hospital"]:
                venue_offered[domain] = "[" + domain + "_name]"

            # if id was not requested but train was found we dont want to override it to check if we booked the right train
            if domain == "train" and (
                not venue_offered[domain] and "id" not in goal["train"]["requestable"]
            ):
                venue_offered[domain] = "[" + domain + "_name]"

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether the right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        # stats 可忽略
        stats = {
            "restaurant": [0, 0, 0],
            "hotel": [0, 0, 0],
            "attraction": [0, 0, 0],
            "train": [0, 0, 0],
            "taxi": [0, 0, 0],
            "hospital": [0, 0, 0],
            "police": [0, 0, 0],
        }

        match = 0
        success = 0
        # MATCH: if the entities provided by the system are correct (剛剛的 db search 如果正確 => MATCH)
        for domain in goal.keys():
            match_stat = 0
            if domain in ["restaurant", "hotel", "attraction", "train"]:
                # goal_venues: Target inform
                # print("Target db result:")
                goal_venues = self.db.queryResultVenues(
                    domain, goal[domain]["informable"], real_belief=True, en=True
                )

                goal_venues_id = []
                if domain == "restaurant":
                    goal_venues_id = [str(venue[0]) for venue in goal_venues]
                elif domain == "attraction":
                    goal_venues_id = [str(venue[0]) for venue in goal_venues]
                elif domain == "train":
                    goal_venues_id = [str(venue[0]).lower() for venue in goal_venues]
                elif domain == "hotel":
                    goal_venues_id = [str(venue[0]) for venue in goal_venues]

                if (
                    type(venue_offered[domain]) is str
                    and "_name" in venue_offered[domain]
                ):
                    match += 1
                    match_stat = 1

                elif len(venue_offered[domain]) > 0:
                    if domain == "restaurant":
                        pred_venues_id = [
                            str(entity[4]) for entity in venue_offered[domain]
                        ]
                        if all([pred in goal_venues_id for pred in pred_venues_id]):
                            match += 1
                            match_stat = 1
                    elif domain == "attraction":
                        pred_venues_id = [
                            str(entity[4]) for entity in venue_offered[domain]
                        ]
                        if all([pred in goal_venues_id for pred in pred_venues_id]):
                            match += 1
                            match_stat = 1
                    elif domain == "train":
                        pred_venues_id = [
                            str(entity[-1]) for entity in venue_offered[domain]
                        ]
                        if all([pred in goal_venues_id for pred in pred_venues_id]):
                            match += 1
                            match_stat = 1
                    elif domain == "hotel":
                        pred_venues_id = [
                            str(entity[5]) for entity in venue_offered[domain]
                        ]
                        if all([pred in goal_venues_id for pred in pred_venues_id]):
                            match += 1
                            match_stat = 1
            else:
                if domain + "_name]" in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match) / len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        # if the entities provided by the system are correct, then calculate the success.
        # SUCCESS: whether the system is able to answer all the requested attributes by user
        # 檢查生成的句子是否有正確提供 user 想要 request 的 slot
        # user 可 request 的 slot 有 requestables: ['name', 'phone', 'address', 'postcode', 'reference', 'id']
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0

                # no request from the user
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue

                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    # request is satisfied
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success) / len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        return success, match, stats

    def _evaluateRealDialogue(self, dialog, filename):
        """Evaluation of the real dialogue.
        First we loads the user goal and then go through the dialogue history.
        Similar to evaluateGeneratedDialogue above."""
        domains = [
            "restaurant",
            "hotel",
            "attraction",
            "train",
            "taxi",
            "hospital",
            "police",
        ]
        requestables = ["phone", "address", "postcode", "reference", "id"]

        # get the list of domains in the goal
        domains_in_goal = []
        goal = {}
        for domain in domains:
            if dialog["goal"][domain]:
                goal = self._parseGoal(goal, dialog, domain)
                domains_in_goal.append(domain)

        # compute corpus success
        venue_offered = {}
        real_requestables = {}
        provided_requestables = {}
        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            # extract domain request
            real_requestables[domain] = goal[domain]["requestable"]

        # iterate each sys turn
        m_targetutt = [
            turn["text"] for idx, turn in enumerate(dialog["log"]) if idx % 2 == 1
        ]
        for t, sent_t in enumerate(m_targetutt):
            for domain in domains_in_goal:
                # for computing match - where there are limited entities
                sent_t = (
                    sent_t.replace("計程車_型別", "taxi_type")
                    .replace("計程車_電話", "taxi_phone")
                    .replace("餐廳_名稱", "restaurant_name")
                )
                if domain + "_name" in sent_t or "_id" in sent_t:
                    if domain in ["restaurant", "hotel", "attraction", "train"]:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        venues = self.db.queryResultVenues(
                            domain, dialog["log"][t * 2 + 1]
                        )

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = random.sample(venues, 1)
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break

                            # sometimes there are no results so sample won't work
                            if not flag and venues:
                                venue_offered[domain] = random.sample(venues, 1)

                    else:  # not limited so we can provide one
                        venue_offered[domain] = "[" + domain + "_name]"

                for requestable in requestables:
                    # check if reference could be issued
                    if requestable == "reference":
                        if domain + "_reference" in sent_t:
                            # if pointer was allowing for that?
                            if "restaurant_reference" in sent_t:
                                # 如果 usr log db_pointer 裡面有 reference 這個 key
                                # ["db_pointer"][-6 ~ -5] 為 [0,1] 表示有 rest_ref，[1,0] 表示沒有
                                if dialog["log"][t * 2]["db_pointer"][-5] == 1:
                                    provided_requestables[domain].append("reference")

                            elif "hotel_reference" in sent_t:
                                # 如果 usr log db_pointer 裡面有 reference 這個 key
                                # ["db_pointer"][-4 ~ -3] 為 [0,1] 表示有 hotel_ref，[1,0] 表示沒有
                                if dialog["log"][t * 2]["db_pointer"][-3] == 1:
                                    provided_requestables[domain].append("reference")

                            elif "train_reference" in sent_t:
                                # 如果 usr log db_pointer 裡面有 reference 這個 key
                                # ["db_pointer"][-2 ~ -1] 為 [0,1] 表示有 train_ref，[1,0] 表示沒有
                                if dialog["log"][t * 2]["db_pointer"][-1] == 1:
                                    provided_requestables[domain].append("reference")

                            else:
                                provided_requestables[domain].append("reference")
                    else:
                        if domain + "_" + requestable in sent_t:
                            provided_requestables[domain].append(requestable)

        # offer was made?
        for domain in domains_in_goal:
            # if name was provided for the user, the match is being done automatically
            if "info" in dialog["goal"][domain]:
                if "name" in dialog["goal"][domain]["info"]:
                    venue_offered[domain] = "[" + domain + "_name]"

            # special domains - entity does not need to be provided
            if domain in ["taxi", "police", "hospital"]:
                venue_offered[domain] = "[" + domain + "_name]"

            # if id was not requested but train was found we dont want to override it to check if we booked the right train
            if domain == "train" and (
                not venue_offered[domain] and "id" not in goal["train"]["requestable"]
            ):
                venue_offered[domain] = "[" + domain + "_name]"

        # HARD (0-1) EVAL
        # [match_stat, success_stat, domain_exist_in_goal]
        # stats 可忽略
        stats = {
            "restaurant": [0, 0, 0],
            "hotel": [0, 0, 0],
            "attraction": [0, 0, 0],
            "train": [0, 0, 0],
            "taxi": [0, 0, 0],
            "hospital": [0, 0, 0],
            "police": [0, 0, 0],
        }

        match, success = 0, 0
        # MATCH: inform, check whether venue_offered in goal_venues or not
        for domain in goal.keys():
            match_stat = 0
            if domain in ["restaurant", "hotel", "attraction", "train"]:
                goal_venues = self.db.queryResultVenues(
                    domain, dialog["goal"][domain]["info"], real_belief=True, en=True
                )

                goal_venues_id = []
                if domain == "restaurant":
                    goal_venues_id = [str(venue[0]) for venue in goal_venues]
                elif domain == "attraction":
                    goal_venues_id = [str(venue[0]) for venue in goal_venues]
                elif domain == "train":
                    goal_venues_id = [str(venue[0]).lower() for venue in goal_venues]
                elif domain == "hotel":
                    goal_venues_id = [str(venue[0]) for venue in goal_venues]

                if (
                    type(venue_offered[domain]) is str
                    and "_name" in venue_offered[domain]
                ):
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0:
                    if (
                        domain == "restaurant"
                        and str(venue_offered[domain][0][4]) in goal_venues_id
                    ):

                        match += 1
                        match_stat = 1
                    elif (
                        domain == "attraction"
                        and str(venue_offered[domain][0][4]) in goal_venues_id
                    ):
                        match += 1
                        match_stat = 1
                    elif (
                        domain == "train"
                        and str(venue_offered[domain][0][-1]).lower() in goal_venues_id
                    ):
                        match += 1
                        match_stat = 1
                    elif (
                        domain == "hotel"
                        and str(venue_offered[domain][0][5]).lower() in goal_venues_id
                    ):
                        match += 1
                        match_stat = 1

            else:
                if domain + "_name" in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if match == len(goal.keys()):
            match = 1
        else:
            match = 0

        # SUCCESS
        if match:
            for domain in domains_in_goal:
                domain_success = 0
                success_stat = 0
                if len(real_requestables[domain]) == 0:
                    # check that
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            # To check whether the sys has fulfilled usr request
            if success >= len(real_requestables):
                success = 1
            else:
                success = 0

        return goal, success, match, real_requestables, venue_offered, stats

    def _parse_entities(self, tokens):
        return [t for t in tokens if "[" in t and "]" in t]

    def evaluateModel_gpt2(self, dialogues, real_dialogues=False, mode="valid"):
        """Gathers statistics for the whole sets."""
        statistics = []
        delex_dialogues = self.delex_dialogues
        successes, matches = 0, 0
        total = 0

        # gen_stats = {
        #     "restaurant": [0, 0, 0],
        #     "hotel": [0, 0, 0],
        #     "attraction": [0, 0, 0],
        #     "train": [0, 0, 0],
        #     "taxi": [0, 0, 0],
        #     "hospital": [0, 0, 0],
        #     "police": [0, 0, 0],
        # }
        # sng_gen_stats = {
        #     "restaurant": [0, 0, 0],
        #     "hotel": [0, 0, 0],
        #     "attraction": [0, 0, 0],
        #     "train": [0, 0, 0],
        #     "taxi": [0, 0, 0],
        #     "hospital": [0, 0, 0],
        #     "police": [0, 0, 0],
        # }

        for idx, (filename, dial) in enumerate(dialogues.items()):
            # if filename == "PMUL4648":

            data = delex_dialogues[filename]

            (
                goal,
                real_success,
                real_match,
                requestables,
                real_venue_offered,
                real_stats,
            ) = self._evaluateRealDialogue(data, filename)

            success, match, stats = self._evaluateGeneratedDialogue(
                filename, dial, goal, data, requestables, soft_acc=mode == "soft"
            )

            successes += success
            matches += match
            total += 1

            statistics.append(filename + "_" + str(match) + "_" + str(success) + "\n")

        if real_dialogues:
            # 計算 BLUE SCORE
            corpus = []
            model_corpus = []
            bscorer = BLEUScorer()

            for dialogue in dialogues:
                data = real_dialogues[dialogue]
                model_turns, corpus_turns = [], []

                for idx, turn in enumerate(data):
                    corpus_turns.append([turn])
                for turn in dialogues[dialogue]["responses"]:
                    model_turns.append([turn])

                if len(model_turns) != len(corpus_turns):
                    raise ("Wrong amount of turns")

                corpus.extend(corpus_turns)
                model_corpus.extend(model_turns)
            model_corpus_len = []
            for turn in model_corpus:
                if turn[0] == "":
                    model_corpus_len.append(True)
                else:
                    model_corpus_len.append(False)
            if all(model_corpus_len):
                print("no model response")
                model_corpus = corpus

            blue_score = bscorer.score(model_corpus, corpus)
        else:
            blue_score = 0.0

        report = ""
        report += (
            "{} Corpus Matches : {:2.2f}%".format(mode, (matches / float(total) * 100))
            + "\n"
        )
        report += (
            "{} Corpus Success : {:2.2f}%".format(
                mode, (successes / float(total) * 100)
            )
            + "\n"
        )
        report += "{} Corpus BLEU : {:2.4f}%".format(mode, blue_score) + "\n"
        report += "Total number of dialogues: %s " % total

        print(report)
        print(
            f"Combined : {(blue_score + 0.5 * (matches / float(total) + successes / float(total))) * 100:.2f}"
        )

        return report, successes / float(total), matches / float(total), statistics


def postprocess_gpt2(generated_raw_data):
    clean_tokens = ["[CLS]", "[SEP]"]
    generated_proc_data = {}
    for key, value in generated_raw_data.items():
        target_beliefs = value["target_turn_belief"]
        target_beliefs_dict = []
        beliefs = value["generated_turn_belief"]
        belief_dict = []

        for turn_id, (turn_target, turn_pred) in enumerate(
            zip(target_beliefs, beliefs)
        ):

            for bs in turn_pred:
                if bs in clean_tokens + ["", " "] or "none" in bs.strip():
                    turn_pred.remove(bs)

            new_turn_pred = []
            for bs in turn_pred:
                for tok in clean_tokens:
                    bs = bs.replace(tok, "").strip()
                new_turn_pred.append(bs)
            turn_pred = new_turn_pred

            # 忽略 value: 未提及
            turn_pred, turn_target = ignore_none(turn_pred, turn_target)

            # MultiWOZ default cleaning
            _, _, bs_dict_pred, bs_dict_target = default_cleaning(
                turn_pred, turn_target, key, turn_id
            )
            belief_dict.append(bs_dict_pred)
            target_beliefs_dict.append(bs_dict_target)

        generated_proc_data[key] = {
            "name": key,
            "responses": value["generated_response"],
            "beliefs": belief_dict,
            # "aggregated_belief": aggregated_belief_dict,
            "target_beliefs": target_beliefs_dict,
            "generated_action": value["generated_action"],
            "target_action": value["target_action"],
        }
    return generated_proc_data


if __name__ == "__main__":
    mode = "test"  # "test" or "soft"
    evaluator = MultiWozEvaluator(mode)
    # 97.40
    # eval_filename = sys.argv[1]
    eval_filename = r"test6endtoend\reinforce_2_ete_0.3_test_context[history=full_history]test_clean.json"
    with open("resources/test_dials.json", "r", encoding="UTF-8") as f:
        human_raw_data = json.load(f)
    human_proc_data = {key: value["sys"] for key, value in human_raw_data.items()}
    with open(eval_filename, "r", encoding="UTF-8") as f:
        generated_raw_data = json.load(f)

    generated_proc_data = postprocess_gpt2(generated_raw_data)

    # PROVIDE HERE YOUR GENERATED DIALOGUES INSTEAD
    generated_data = generated_proc_data

    report, success, match, statistics = evaluator.evaluateModel_gpt2(
        generated_data, human_proc_data, mode=mode
    )

    with open("e2e03.txt", "wt", encoding="UTF-8") as f:
        f.writelines(statistics)
