# models=("/allenai/longformer-base-4096" "/microsoft/deberta-xlarge-mnli")
# models=( "/Qwen/Qwen2-7B-Instruct" "/microsoft/Phi-3-small-128k-instruct" )
# models=( "/microsoft/Phi-3-small-128k-instruct" "/allenai/primera-multi_lexsum-source-short")
models=( "/allenai/primera-multi_lexsum-source-short" "/allenai/primera-multi_lexsum-source-tiny" "/allenai/primera-multi_lexsum-source-long" "/allenai/led-base-16384-multi_lexsum-source-long" "/allenai/led-base-16384-multi_lexsum-source-short" "/allenai/led-base-16384-multi_lexsum-source-tiny")
path="models/"

if [ ! -d $path ]; then
    echo "$path does not exist; creating path"
    mkdir $path
fi
cd $path
 
for model_path in ${models[@]}; do
    readarray -d "/" -t names <<< "$model_path"
    model_name=${names[-1]} 
    if ! [ -d $model_name ]; then
        git clone https://huggingface.co$model_path
    else
        echo $model_name has already been downloaded
    fi
done