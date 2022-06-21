#sh scripts/gen_feature-2018.sh
#sh scripts/gen_label-2018.sh
for((i=1;i<=1;i++))
do
CUDA_VISIBLE_DEVICES=2 python main.py -n DCASE2018-task4 -s sed_with_cATP-DF -u false -md train -e false -g false
mv exp/DCASE2018-task4 exp/DCASE2018-task4_$i
done
