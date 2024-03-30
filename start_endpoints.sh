python -m llama_cpp.server \
        --model model/yarn-mistral-7b-128k.Q4_K_M.gguf \
        --chat_format chatml \
        --model_alias gpt-3.5-turbo\
        --n_gpu_layers -1\
