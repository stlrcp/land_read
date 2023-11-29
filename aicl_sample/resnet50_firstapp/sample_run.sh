model_name="MyFirstApp_run"
APP_SOURCE_PATH="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"

cd ${APP_SOURCE_PATH}/out

./main
