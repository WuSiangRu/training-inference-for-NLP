from utils.Constants import SLOT_VALS
from utils.dst import ignore_none, default_cleaning, IGNORE_TURNS_TYPE2
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--eval_file", default=str, help="evaluate file name (json)")
parser.add_argument("--task", default=str, help="original or dstc9")
parser.add_argument(
    "--default_cleaning", action="store_true", help="use default cleaning from multiwoz"
)

args = parser.parse_args()
# args.eval_file = (
# r"generated_output\dst_no_special_test_context[history=full_history]test_clean_4629.json"
# )
args.eval_file = (
   # r"../dst_output_test_context[history=full_history]_test_clean_5013.json"
    r"simpletod_specialadded_rl_04_none_baseline_test_context[history=full_history]test_clean.json"
)
args.task = "original"
args.default_cleaning = True

data = json.load(open(args.eval_file, "r", encoding="UTF-8"))

# with open("below_jga_ignore_nmdc_dsct9.txt") as f:
#     ignore_dialogues = f.readlines()

# ignore = [line.split()[0] for line in ignore_dialogues]


num_turns = 0
joint_acc = 0
soft_acc = 0

# clean_tokens = ['<|endoftext|>']
clean_tokens = ["[CLS]", "[SEP]"]

# store jga < 0.5 dial
below_jga = []

for dial in data:
    # if dial == "PMUL4125":
    dialogue_pred = data[dial]["generated_turn_belief"]
    dialogue_target = data[dial]["target_turn_belief"]
    model_context = data[dial]["model_context"]

    dial_turn = 0
    dial_acc = 0

    for turn_id, (turn_target, turn_pred, turn_context) in enumerate(
        zip(dialogue_target, dialogue_pred, model_context)
    ):

        dial_turn += 1
        # print("turn_pred", turn_pred)
        # print("turn_target", turn_target)
        # clean
        for bs in turn_pred:
            # if bs in clean_tokens + ["", " "] or bs.split()[-1] == "none":
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
###mod_by_samuel
        #print("args.task", args.task)
        if args.default_cleaning:
            turn_pred, turn_target, bs_dict_pred, bs_dict_target = default_cleaning(
                turn_pred, turn_target, dial, turn_id, task=args.task
            )
        # print(bs_dict_pred)
        # print(bs_dict_target)
        # join_flag = False
        pred = sorted(set(turn_pred))
        target = sorted(set(turn_target))

        if target == pred:
            joint_acc += 1
            dial_acc += 1
            # join_flag = True

        ac = len([t for t in target if t in pred]) / (len(pred) + 1e-8)
        soft_acc += ac

        num_turns += 1

    # dial_acc /= dial_turn
    # if 0.3 < dial_acc < 0.4:
    #     below_jga.append(dial + f" {dial_acc:.2f}\n")


joint_acc /= num_turns
soft_acc /= num_turns

#print("joint accuracy: {}".format(joint_acc))
print("jont accuracy:", "%.30f"%joint_acc)
print("soft accuracy: {}".format(soft_acc))

#%%
# with open("below_jga_ignore_nmdc_dsct9_dst0304.txt", "w", encoding="UTF-8") as f:
#     f.writelines(below_jga)
# %%
