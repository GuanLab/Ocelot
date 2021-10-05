name1=MODEL_TEMPLATE

for i in {01..NUM}
do
    echo $i
    cd ${name1}_${i}
    mkdir -p model
    #time ./bash.sh &
    time ./bash.sh 
    cd ../
    sleep 10s
done
wait

