# Код к презентации "Как начать экспериментировать с машинным обучением в Go"

## Гофер повторяшка 

```shell script
source setupvars.sh
go run -tags openvino dnn-pose-detection/*.go
```

## Gorgonia + CUDA GPU 

```shell script
cd gorgonia
docker-compose build && docker-compose run project
```

## Загрузка моделей OpenVINO

https://software.intel.com/en-us/articles/model-downloader-essentials