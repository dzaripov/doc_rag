## Последовательность действий по запуску сервиса

0. Сохраняем репозиторий локально:
`git clone https://github.com/dzaripov/doc_rag`

2. Поднимаем milvus
`docker-compose up -d`

3. Запускаем API
`uvicorn main:app --reload`

4. Запускаем интерфейс
`python demo.py`

Сервис будет доступен по адресу, указанному в логах запуска интерфейса
