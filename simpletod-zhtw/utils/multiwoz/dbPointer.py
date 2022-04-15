import sqlite3

import numpy as np

from .nlp import normalize
import os

PATH = "./utils/multiwoz"

# loading databases
domains = [
    "restaurant",
    "hotel",
    "attraction",
    "train",
    "taxi",
    "hospital",
]  # , 'police']

dbs = {}
for domain in domains:
    db = os.path.join(PATH, "db/{}-dbase.db".format(domain))
    conn = sqlite3.connect(db)
    c = conn.cursor()
    dbs[domain] = c

dbs_dstc9 = {}
for domain in domains:
    # db = os.path.join("utils/multiwoz/db/{}-dbase.db".format(domain))
    db = os.path.join("utils/multiwoz/db/zhtw_db/zhtw_{}_dbase.db".format(domain))
    conn = sqlite3.connect(db)
    c = conn.cursor()
    dbs_dstc9[domain] = c

domains_map = {
    "restaurant": "餐廳",
    "hotel": "旅館",
    "attraction": "景點",
    "train": "列車",
    "taxi": "計程車",
    "hospital": "醫院",
    "police": "警察機關",
}


def convert_dbpointer_to_text(vect, goal, belief):
    # vect: db_data[i]，第 i 輪 sys 的 db
    # goal: 該json對話的 goal
    # belief: d_lex['belief_raw'][i]，第 i 輪 sys 的 belief_raw，資料格式為 [domain, slot, value]

    en_domain_in_pointer = ["restaurant", "hotel", "attraction", "train"]
    tw_domain_in_pointer = ["餐廳", "旅館", "景點", "列車"]
    # domains = ['餐廳', '旅館', '景點', '列車', '計程車', '醫院', '警察機關']
    restaurant_book_vec = vect[24:26]
    hotel_book_vec = vect[26:28]
    train_book_vec = vect[28:]
    text = []
    for idx, (en_domain, tw_domain) in enumerate(
        zip(en_domain_in_pointer, tw_domain_in_pointer)
    ):
        if en_domain not in goal:
            continue

        Flag = any(bs[0] == tw_domain for bs in belief)
        if not Flag:  # not bstate for domain
            continue

        domain_vec = vect[idx * 6 : idx * 6 + 6]
        if en_domain != "train":  # restaurant, hotel, attraction domains
            if np.all(domain_vec == np.array([1, 0, 0, 0, 0, 0])):
                domain_match = 0
            elif np.all(domain_vec == np.array([0, 1, 0, 0, 0, 0])):
                domain_match = 1
            elif np.all(domain_vec == np.array([0, 0, 1, 0, 0, 0])):
                domain_match = 2
            elif np.all(domain_vec == np.array([0, 0, 0, 1, 0, 0])):
                domain_match = 3
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 1, 0])):
                domain_match = 4
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 0, 1])):
                domain_match = 5
            else:
                raise ValueError("invalid domain match")

            domain_match_text = (
                ">=5" if domain_match >= 5 else "={}".format(domain_match)
            )
            if (
                en_domain == "restaurant"
                and np.all(restaurant_book_vec == np.array([0, 1]))
            ) or (en_domain == "hotel" and np.all(hotel_book_vec == np.array([0, 1]))):
                # 如果 [metadata][restaurant/hotel][book][booked] 存在 reference 這個 key，則該domain的vec == [0, 1]
                text.append("{} 符合{} 預定=available".format(tw_domain, domain_match_text))

            else:
                text.append(
                    "{} 符合{} 預定=not available".format(tw_domain, domain_match_text)
                )

        else:  # train domain
            if np.all(domain_vec == np.array([1, 0, 0, 0, 0, 0])):
                domain_match = 0
            elif np.all(domain_vec == np.array([0, 1, 0, 0, 0, 0])):
                domain_match = 2
            elif np.all(domain_vec == np.array([0, 0, 1, 0, 0, 0])):
                domain_match = 5
            elif np.all(domain_vec == np.array([0, 0, 0, 1, 0, 0])):
                domain_match = 10
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 1, 0])):
                domain_match = 40
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 0, 1])):
                domain_match = 41
            else:
                raise ValueError("invalid domain match")

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

            if np.all(train_book_vec == np.array([0, 1])):
                # 如果 [metadata][train][book][booked] 存在 reference 這個 key，則該domain的vec == [0, 1]
                text.append("{} 符合{} 預定=available".format(tw_domain, domain_match_text))
            else:
                text.append(
                    "{} 符合{} 預定=not available".format(tw_domain, domain_match_text)
                )

    return " , ".join(text)


def convert_dbpointer_to_text_nmatch(vect, goal, belief):
    # vect: db_data[i]，第 i 輪 sys 的 db
    # goal: 該json對話的 goal
    # belief: d_lex['belief_raw'][i]，第 i 輪 sys 的 belief_raw，資料格式為 [domain, slot, value]
    en_domain_in_pointer = ["restaurant", "hotel", "attraction", "train"]
    tw_domain_in_pointer = ["餐廳", "旅館", "景點", "列車"]
    # domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']#, 'police']
    restaurant_book_vec = vect[24:26]  # res_vec
    hotel_book_vec = vect[26:28]  # hotel_vec
    train_book_vec = vect[28:]  # train_vec
    text = []
    for idx, (en_domain, tw_domain) in enumerate(
        zip(en_domain_in_pointer, tw_domain_in_pointer)
    ):
        if en_domain not in goal:
            continue

        # Flag == False when all bs[0] != domain
        Flag = any(bs[0] == tw_domain for bs in belief)
        if not Flag:  # not bstate for domain
            continue

        domain_vec = vect[idx * 6 : idx * 6 + 6]
        if en_domain != "train":
            if np.all(domain_vec == np.array([1, 0, 0, 0, 0, 0])):
                domain_match = 0
            elif np.all(domain_vec == np.array([0, 1, 0, 0, 0, 0])):
                domain_match = 1
            elif np.all(domain_vec == np.array([0, 0, 1, 0, 0, 0])):
                domain_match = 2
            elif np.all(domain_vec == np.array([0, 0, 0, 1, 0, 0])):
                domain_match = 3
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 1, 0])):
                domain_match = 4
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 0, 1])):
                domain_match = 5
            else:
                raise ValueError("invalid domain match")

            domain_match_text = (
                ">=5" if domain_match >= 5 else "={}".format(domain_match)
            )

        else:  # train domain
            if np.all(domain_vec == np.array([1, 0, 0, 0, 0, 0])):
                domain_match = 0
            elif np.all(domain_vec == np.array([0, 1, 0, 0, 0, 0])):
                domain_match = 2
            elif np.all(domain_vec == np.array([0, 0, 1, 0, 0, 0])):
                domain_match = 5
            elif np.all(domain_vec == np.array([0, 0, 0, 1, 0, 0])):
                domain_match = 10
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 1, 0])):
                domain_match = 40
            elif np.all(domain_vec == np.array([0, 0, 0, 0, 0, 1])):
                domain_match = 41
            else:
                raise ValueError("invalid domain match")

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

        text.append("{} 符合{}".format(tw_domain, domain_match_text))

    return " , ".join(text)


def oneHotVector(num, domain, vector):
    """Return number of available entities for particular domain."""
    # num -> int : length of query results
    # domain -> string
    # vector -> zero's array of length 24

    domains = [
        "restaurant",
        "hotel",
        "attraction",
        "train",
        "taxi",
        "hospital",
    ]  # , 'police']
    # number_of_options = 6
    if domain != "train":
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6 : idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num == 1:
            vector[idx * 6 : idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num == 2:
            vector[idx * 6 : idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num == 3:
            vector[idx * 6 : idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num == 4:
            vector[idx * 6 : idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num >= 5:
            vector[idx * 6 : idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
    else:
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6 : idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num <= 2:
            vector[idx * 6 : idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num <= 5:
            vector[idx * 6 : idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num <= 10:
            vector[idx * 6 : idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num <= 40:
            vector[idx * 6 : idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num > 40:
            vector[idx * 6 : idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])

    return vector


def queryResult(domain, turn):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""
    # query the db
    # turn: sys turn
    sql_query = "select * from {}".format(domain)

    flag = True
    # print turn['metadata'][domain]['semi']
    for key, val in turn["metadata"][domain]["semi"].items():
        if (
            val == ""
            or val == "dont care"
            or val == "not mentioned"
            or val == "don't care"
            or val == "dontcare"
            or val == "do n't care"
        ):
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                # val2 = normalize(val2)
                # change query for trains
                if key == "leaveAt":
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == "arriveBy":
                    sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                # val2 = normalize(val2)
                if key == "leaveAt":
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == "arriveBy":
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    # try:  # "select * from attraction  where name = 'queens college'"
    # print sql_query
    # print domain
    num_entities = len(dbs[domain].execute(sql_query).fetchall())

    return num_entities


def queryResultDSTC9(domain, turn):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""
    tables_map = {
        "restaurant": "RESTAURANTS",
        "hotel": "HOTELS",
        "attraction": "ATTRACTIONS",
        "train": "TRAINS",
    }

    # query the db
    # turn: sys turn
    table = tables_map[domain]
    sql_query = "select * from {}".format(table)

    flag = True
    # print turn['metadata'][domain]['semi']
    for key, val in turn["metadata"][domains_map[domain]]["semi"].items():
        if (
            val == ""
            or val == "dont care"
            or val == "not mentioned"
            or val == "don't care"
            or val == "dontcare"
            or val == "do n't care"
            or val == "不在意"
        ):
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                # val2 = normalize(val2)
                # change query for trains
                if key == "leaveAt":
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == "arriveBy":
                    sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                # val2 = normalize(val2)
                if key == "leaveAt":
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == "arriveBy":
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    # try:  # "select * from attraction  where name = 'queens college'"
    # print sql_query
    # print domain
    num_entities = len(dbs_dstc9[domain].execute(sql_query).fetchall())

    return num_entities


def queryResultVenues(domain, turn, real_belief=False):
    # query the db
    sql_query = "select * from {}".format(domain)

    if real_belief == True:
        items = turn.items()
    elif real_belief == "tracking":
        for slot in turn[domain]:
            key = slot[0].split("-")[1]
            val = slot[0].split("-")[2]
            if key == "price range":
                key = "pricerange"
            elif key == "leave at":
                key = "leaveAt"
            elif key == "arrive by":
                key = "arriveBy"
            if val == "do n't care":
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == "leaveAt":
                        sql_query += key + " > " + r"'" + val2 + r"'"
                    elif key == "arriveBy":
                        sql_query += key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key == "leaveAt":
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    elif key == "arriveBy":
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

            try:  # "select * from attraction  where name = 'queens college'"
                return dbs[domain].execute(sql_query).fetchall()
            except:
                return []  # TODO test it
        pass
    else:
        items = turn["metadata"][domain]["semi"].items()

    flag = True
    for key, val in items:
        if (
            val == ""
            or val == "dontcare"
            or val == "not mentioned"
            or val == "don't care"
            or val == "dont care"
            or val == "do n't care"
        ):
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                if key == "leaveAt":
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == "arriveBy":
                    sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                val2 = normalize(val2)
                if key == "leaveAt":
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == "arriveBy":
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    try:  # "select * from attraction  where name = 'queens college'"
        return dbs[domain].execute(sql_query).fetchall()
    except:
        return []  # TODO test it
