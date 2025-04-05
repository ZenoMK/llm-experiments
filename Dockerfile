FROM python:3.12

#WORKDIR llm-experiments

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash", "run.sh"]