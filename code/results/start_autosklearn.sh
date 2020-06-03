
echo "$1 algo"
count=10
for target in "lr_acc" "lr_auroc" "knn_pca_acc" "knn_pca_auroc" "rf_pca_acc" "rf_pca_auroc"
do
  for dataset in "birth_randoms" "ring_randoms" "adult_randoms" "heart_randoms"
  do
    if (($count == 21));
    then
      count=$((count+1))
    fi
    echo "$(printf %02d $count)corona"
    ssh rd2016@corona$(printf %02d $count).doc.ic.ac.uk "cd Entropy_utility_measure/code/results; nohup python3 auto_tune.py ${dataset} ${target} $1> ${dataset}_${target}_$1.out & "
    count=$((count+1))
  done
done
