
run_name='director'
notes='baseline'
echo "Run name: $run_name $notes"
for i in {1..4}; do
    echo "**************Run $i***************"
    python dreamer.py --configs pinpad_four --logdir ./logs/pinpad/$run_name/$notes/$(date +%s) --demodir /home/j/workspace/fastrl/logs/HD_pinpad_four_all/0 --train_ratio 256 --pretrain 5000 --reduce_lr_after_pretraining False
    sleep 1
done