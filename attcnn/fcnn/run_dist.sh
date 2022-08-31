
for i in {1..10..1}
do
    python dist_train.py --limit 100 --mixed 0 --eps 100 --seed $i 
    python dist_test.py --limit 100 --mixed 0 --seed $i 
done


for i in {1..10..1}
do
    python dist_train.py --limit 100 --mixed 1 --eps 100 --seed $i 
    python dist_test.py --limit 100 --mixed 1 --seed $i 
done


for i in {1..10..1}
do
    python dist_train.py --limit 100 --mixed 0 --eps 100 --seed $i 
    python dist_test.py --limit 100 --mixed 1 --seed $i 
done