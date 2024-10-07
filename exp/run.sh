cd ../

mkdir -p Logs

# User Setting Parameter
SERVER=8
CONCEPT="cl"
MODEL="vonet"
TEST_DATA_NAME="kitti"


# Exp folder setting
EXP_NAME="Server_${SERVER}_${CONCEPT}_${MODEL}_test_${TEST_DATA_NAME}"
DIR="result/${EXP_NAME}"
i=1
while [ -d "$DIR" ]; do
    DIR="result/${EXP_NAME}_${i}"
    i=$((i+1))
done

# Epoch, Round Setting(option)
if [ "$CONCEPT" = "fl" ]; then
    NODE_NUM=18
    MODEL_PATH="models/exp_posenet_NODE18_ITER3_both.pth"
    EPOCH=3
    if [ "$MODEL" = "vonet" ]; then #at paper, use 50,000 iter -> 10 round
        ROUND=10
        BATCH=64
    elif [ "$MODEL" = "posenet" ]; then #at paper, use 100,000 iter -> 30 round
        ROUND=30
        BATCH=100
    fi
else
    NODE_NUM=1
    MODEL_PATH="models/exp_posenet_NODE1_ITER1_both.pth"
    EPOCH=1
    if [ "$MODEL" = "vonet" ]; then #at paper, use 50,000 iter -> 10 round
        ROUND=10
        BATCH=64
    elif [ "$MODEL" = "posenet" ]; then #at paper, use 100,000 iter -> 30 round
        ROUND=30
        BATCH=100
    fi
fi

# Dataset path setting
if [ $SERVER = 5 ]; then
    TRAIN_DATA_PATH="/Dataset/tartanAir/train_data"
elif [ $SERVER = 8 ]; then
    TRAIN_DATA_PATH="/scratch/jeongeon/tartanAir/train_data"
fi
if [ "$TEST_DATA_NAME" = "kitti" ]; then
    TEST_DATA_PATH="data/KITTI_10"
    DATA_MODE="basic"
elif [ "$TEST_DATA_NAME" = "euroc" ]; then
    TEST_DATA_PATH="data/EuRoC_V102"
    DATA_MODE="basic"
else
    TEST_DATA_PATH=$TRAIN_DATA_PATH
    DATA_MODE="all"
fi

# Run Code
python3 -u main.py -node $NODE_NUM -model "$MODEL" -model_path "$MODEL_PATH" -train_path "$TRAIN_DATA_PATH" \
                -test_dataset "$TEST_DATA_NAME" -test_path "$TEST_DATA_PATH" \
                -batch $BATCH -epoch $EPOCH -round $ROUND -data_mode $DATA_MODE 1> Logs/${EXP_NAME}.log 2>&1 

wait
echo "Experiments completed!"
