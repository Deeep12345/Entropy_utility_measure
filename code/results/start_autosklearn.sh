

count=10
for target in "lr_acc" "lr_auroc" "knn_pca_acc" "knn_pca_auroc" "rf_pca_acc" "rf_pca_auroc"
do
  for dataset in "birth_randoms" "ring_randoms" "adult_randoms" "heart_randoms"
  do
    echo $count
    ssh rd2016@corona${count}.doc.ic.ac.uk "skill -u rd2016; cd Entropy_utility_measure/code/results; nohup python3 auto_tune.py ${dataset} ${target} > ${dataset}_${target}.out & "
    count=$((count+1))
  done
done
