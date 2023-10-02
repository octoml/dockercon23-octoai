FROM python:3.11.4
RUN pip install discord Pillow requests octoai-sdk
WORKDIR /code
COPY workflow.py /code
COPY bot.py /code
CMD python bot.py
