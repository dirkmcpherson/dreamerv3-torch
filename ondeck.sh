# run n times

# for i in {1..5}; do 
#     python dreamer.py --configs pinpad_four --logdir ./logs/pinpad/$(date +%s) --demodir /home/j/workspace/fastrl/logs/HD_pinpad_four_all/0
#     sleep 1
# done


# for i in {1..5}; do 
#     python dreamer.py --configs pinpad_four --logdir ./logs/pinpad/$(date +%s) --demodir /home/j/workspace/fastrl/logs/HD_pinpad_four_all/0 --pretrain 1000 --train_ratio 256
#     sleep 1
# done

run_name='pretrain5k'
notes='lr_reduce'
echo "Run name: $run_name $notes"
for i in {1..2}; do
    echo "**************Run $i***************"
    python dreamer.py --configs pinpad_four --logdir ./logs/pinpad/$run_name/$notes/$(date +%s) --demodir /home/j/workspace/fastrl/logs/HD_pinpad_four_all/0 --train_ratio 256 --pretrain 5000 --reduce_lr_after_pretraining True
    sleep 1
done

notes='no_lr_reduce'
echo "Run name: $run_name $notes"
for i in {1..2}; do
    echo "**************Run $i***************"
    python dreamer.py --configs pinpad_four --logdir ./logs/pinpad/$run_name/$notes/$(date +%s) --demodir /home/j/workspace/fastrl/logs/HD_pinpad_four_all/0 --train_ratio 256 --pretrain 5000 --reduce_lr_after_pretraining False
    sleep 1
done

# run_name='baseline'
# echo "Run name: $run_name"
# for i in {1..4}; do
#     echo "**************Run $i***************"
#     python dreamer.py --configs pinpad_four --logdir ./logs/pinpad/$run_name/$notes/$(date +%s) --demodir /home/j/workspace/fastrl/logs/HD_pinpad_four_all/0 --train_ratio 256
#     sleep 1
# done

# run_name='vanilla'
# echo "Run name: $run_name"
# for i in {1..4}; do
#     echo "**************Run $i***************"
#     python dreamer.py --configs pinpad_four --logdir ./logs/pinpad/$run_name/$notes/$(date +%s) --train_ratio 256
#     sleep 1
# done