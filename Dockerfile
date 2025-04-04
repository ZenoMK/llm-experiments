FROM python:3.12

WORKDIR /Users/zenokujawa/PycharmProjects/llm-experiments

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "data/circle/create_circle.py --num_nodes 100"]