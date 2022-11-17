FROM python:3.9

# Location of source code
ENV PROJECT_ROOT /opt/project
RUN mkdir -p $PROJECT_ROOT
WORKDIR $PROJECT_ROOT

# Copying dependencies
COPY ./requirements.txt .

RUN pip install setuptools wheel
RUN pip install -r requirements.txt

#WORKDIR /root
#COPY . .
COPY ./src ./src
COPY ./dataset ./dataset

RUN pip install --upgrade pip
RUN apt-get update -y && \
    apt-get install -y vim curl && \
    rm -rf var/lib/apt/lists/* \

