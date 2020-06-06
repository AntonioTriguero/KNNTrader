FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

CMD [ "python", "-u", "./server.py" ]
