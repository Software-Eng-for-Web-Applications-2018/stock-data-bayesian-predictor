#! /bin/sh 
tensorflow_model_server --port=9040 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_RT_AABA &
tensorflow_model_server --port=9041 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_RT_AAPL &
tensorflow_model_server --port=9042 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_RT_AMD &
tensorflow_model_server --port=9043 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_RT_AMZN &
tensorflow_model_server --port=9044 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_RT_C &
tensorflow_model_server --port=9045 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_RT_GOOG &
tensorflow_model_server --port=9046 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_RT_GOOGL &
tensorflow_model_server --port=9047 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_RT_INTC &
tensorflow_model_server --port=9048 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_RT_MSFT &
tensorflow_model_server --port=9049 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_RT_VZ &
#Historical
tensorflow_model_server --port=9050 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_PAST_AABA &
tensorflow_model_server --port=9051 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_PAST_AAPL &
tensorflow_model_server --port=9052 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_PAST_AMD &
tensorflow_model_server --port=9053 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_PAST_AMZN &
tensorflow_model_server --port=9054 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_PAST_C &
tensorflow_model_server --port=9055 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_PAST_GOOG &
tensorflow_model_server --port=9056 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_PAST_GOOGL &
tensorflow_model_server --port=9057 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_PAST_INTC &
tensorflow_model_server --port=9058 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_PAST_MSFT &
tensorflow_model_server --port=9059 --model_name=BAYMODEL --model_base_path=$(pwd)/BAY_PAST_VZ &
