import re
import os
import simplejson as json
import sys
from .nlp import normalize

import ipdb

digitpat = re.compile("\d+")
timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat2 = re.compile("\d{1,3}[.]\d{1,2}")

# FORMAT
# domain_value
# restaurant_postcode
# restaurant_address
# taxi_car8
# taxi_number
# train_id etc..


def prepareSlotValuesIndependent():
    en_domains = [
        "restaurant",
        "hotel",
        "attraction",
        "train",
        "taxi",
        "hospital",
        "police",
    ]
    # tw_domains = ["餐廳", "旅館", "景點", "列車", "計程車", "醫院", "警察機關"]

    # requestables = ['phone', 'address', 'postcode', 'reference', 'id']
    dic = []  # dic.append((normalize(slot_value), [domain_slot]))
    # dic = address * name * postcode * phone * trainID * department * hospital * police *
    #       departure or destination * monday... * area * food * pricerange
    dic_area = []
    dic_food = []
    dic_price = []
    # read databases
    PATH = r"resources/multi-woz/MULTIWOZ2.1"
    for domain in en_domains:
        try:
            # fin = file(os.path.join(PATH, 'db/' + domain + '_db.json'))
            fin = open(os.path.join(PATH, f"zhtw_{domain}_db.json"), encoding="UTF-8")
            db_json = json.load(fin)
            fin.close()

            for ent in db_json:
                for key, val in ent.items():
                    if val == "?" or val == "free":
                        pass
                    elif key == "地址":
                        dic.append(
                            (normalize(val), "[" + domain + "_" + "address" + "]")
                        )
                        # if "road" in val:
                        #     val = val.replace("road", "rd")
                        #     dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        # elif "rd" in val:
                        #     val = val.replace("rd", "road")
                        #     dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        # elif "st" in val:
                        #     val = val.replace("st", "street")
                        #     dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                        # elif "street" in val:
                        #     val = val.replace("street", "st")
                        #     dic.append((normalize(val), '[' + domain + '_' + 'address' + ']'))
                    elif key == "名稱":
                        dic.append((normalize(val), "[" + domain + "_" + "name" + "]"))
                        # if "b & b" in val:
                        #     val = val.replace("b & b", "bed and breakfast")
                        #     dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        # elif "bed and breakfast" in val:
                        #     val = val.replace("bed and breakfast", "b & b")
                        #     dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        # elif "hotel" in val and 'gonville' not in val:
                        #     val = val.replace("hotel", "")
                        #     dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                        # elif "restaurant" in val:
                        #     val = val.replace("restaurant", "")
                        #     dic.append((normalize(val), '[' + domain + '_' + 'name' + ']'))
                    elif key == "郵編":
                        dic.append(
                            (normalize(val), "[" + domain + "_" + "postcode" + "]")
                        )
                    elif key == "電話":
                        dic.append((val, "[" + domain + "_" + "phone" + "]"))
                    elif key == "列車號":
                        dic.append((normalize(val), "[" + domain + "_" + "id" + "]"))
                    elif key == "科室":
                        dic.append(
                            (normalize(val), "[" + domain + "_" + "department" + "]")
                        )

                    # NORMAL DELEX
                    elif key == "區域":
                        dic_area.append(
                            (normalize(val), "[" + "value" + "_" + "area" + "]")
                        )
                        if val in ["北方", "南方", "東方", "西方"]:
                            dic_area.append(
                                (val[0] + "部", "[" + "value" + "_" + "area" + "]")
                            )
                            dic_area.append(
                                (val[0] + "側", "[" + "value" + "_" + "area" + "]")
                            )

                    elif key == "食物":
                        dic_food.append(
                            (normalize(val), "[" + "value" + "_" + "food" + "]")
                        )
                    elif key == "價格範圍":
                        dic_price.append(
                            (normalize(val), "[" + "value" + "_" + "pricerange" + "]")
                        )
                        if "自由" in val:
                            dic_price.append(
                                ("自由", "[" + "value" + "_" + "pricerange" + "]")
                            )
                            dic_price.append(
                                ("自由的", "[" + "value" + "_" + "pricerange" + "]")
                            )
                            dic_price.append(
                                ("免費", "[" + "value" + "_" + "pricerange" + "]")
                            )
                            dic_price.append(
                                ("免費的", "[" + "value" + "_" + "pricerange" + "]")
                            )
                        if "價格" in val:
                            dic_price.append(
                                (val[2:], "[" + "value" + "_" + "pricerange" + "]")
                            )
                    else:
                        # TODO car type?
                        pass

        except:
            pass

        if domain == "hospital":
            dic.append(("希爾斯路", "[" + domain + "_" + "address" + "]"))
            dic.append((normalize("CB20QQ"), "[" + domain + "_" + "postcode" + "]"))
            dic.append(("01223245151", "[" + domain + "_" + "phone" + "]"))
            dic.append(("1223245151", "[" + domain + "_" + "phone" + "]"))
            dic.append(("0122324515", "[" + domain + "_" + "phone" + "]"))
            dic.append(("阿登布魯克醫院", "[" + domain + "_" + "name" + "]"))

        elif domain == "police":
            dic.append(("parkside", "[" + domain + "_" + "address" + "]"))
            dic.append(("劍橋公園邊", "[" + domain + "_" + "address" + "]"))
            dic.append((normalize("CB11JG"), "[" + domain + "_" + "postcode" + "]"))
            dic.append(("01223358966", "[" + domain + "_" + "phone" + "]"))
            dic.append(("1223358966", "[" + domain + "_" + "phone" + "]"))
            dic.append(("帕克賽德警察局", "[" + domain + "_" + "name" + "]"))

    # add at the end places from trains
    # fin = open(os.path.join(PATH, 'db/' + 'train' + '_db.json'))
    fin = open(os.path.join(PATH, f"zhtw_train_db.json"), encoding="UTF-8")
    db_json = json.load(fin)
    fin.close()

    for ent in db_json:
        for key, val in ent.items():
            if key == "出發地" or key == "目的地":
                dic.append((normalize(val), "[" + "value" + "_" + "place" + "]"))

    # add specific values:
    for key in ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日", "星期天"]:
        dic.append((normalize(key), "[" + "value" + "_" + "day" + "]"))

    for key in ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]:
        dic.append((normalize(key), "[" + "value" + "_" + "day" + "]"))

    # more general values add at the end
    dic.extend(dic_area)
    dic.extend(dic_food)
    dic.extend(dic_price)

    return dic


def delexicalise(utt, dictionary):
    for key, val in dictionary:
        # utt = (" " + utt + " ").replace(" " + key + " ", " " + val + " ")
        utt = utt.replace(key, val)
        # utt = utt[1:-1]  # why this?

    return utt


def delexicaliseDomain(utt, dictionary, domain):
    for key, val in dictionary:
        if key in [domain, "value"]:
            utt = (" " + utt + " ").replace(" " + key + " ", " " + val + " ")
            utt = utt[1:-1]  # why this?

    # go through rest of domain in case we are missing something out?
    for key, val in dictionary:
        utt = (" " + utt + " ").replace(" " + key + " ", " " + val + " ")
        utt = utt[1:-1]  # why this?
    return utt


if __name__ == "__main__":
    prepareSlotValuesIndependent()
