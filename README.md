# training-inference-for-NLP
training&amp;inference in multidomain chatbot with trainer

##### 檔案分類說明
1. run_clm_custom_trainer_zhtw_mod.py (中文有兩個演算法的使用RL)
2. run_clm_custom_trainer_zhtw_mod_rdrop.py (中文有三個演算法的使用R-Drop)
3. run_clm_custom_trainer_mod.py (英文有兩個演算法的使用RL)
4. run_clm_custom_trainer_mod_rdrop.py (英文有三個演算法的使用R-Drop)

以上皆有使用Trainer訓練

---------------

_*資料夾有區分中文及英文*_
* simpletod-eng (英文的)
* simpletod-zhtw (中文的)

`其中"replace_bs_none_action_repeat訓練end2end之前請先處理重複的action.py"是在訓練中文end-to-end實驗前要對資料集使用的`