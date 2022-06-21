#sh scripts/gen_feature-2018.sh
#sh scripts/gen_label-2018.sh
for((i=1;i<=1;i++))
do
CUDA_VISIBLE_DEVICES=1 python main.py -n DCASE2019-task4 -s sed_with_cATP-DF -t at_with_cATP-DF  -u true -md tune -e false -g false
#mv exp/DCASE2019-task4 exp/DCASE2019-task4_$i
done
