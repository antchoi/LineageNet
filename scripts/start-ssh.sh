#!/bin/bash
SCRIPT_DIR=$(dirname $(readlink -f $0))
ROOT_DIR="${SCRIPT_DIR}/.."
ENV_FILE="${ROOT_DIR}/.env"
TMP_DIR="${ROOT_DIR}/tmp"

lines=$(grep -v '^#' ${ENV_FILE})
IFS=$'\n' read -rd '' -a array <<<"${lines}"

for line in "${array[@]}"
do
    if [[ -z ${line} ]]; then
        continue
    fi
    export "${line}"
done

mkdir -p ${TMP_DIR}

if [ -f "/usr/lib/x86_64-linux-gnu/libnvcuvid.so" ]; then
    cp /usr/lib/x86_64-linux-gnu/libnvcuvid.so ${TMP_DIR}/libnvcuvid.so
fi

export USER_ID=$(id -u)
export USER_NAME=$(id -un)
export GROUP_ID=$(id -g)
export GROUP_NAME=$(id -gn)

pushd ${ROOT_DIR} && {
    COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker-compose -f docker-compose-ssh.yml up -d --remove-orphans ${1}
} && popd
