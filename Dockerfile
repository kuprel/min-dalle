FROM nvcr.io/nvidia/pytorch:22.04-py3

ARG WORKDIR=/min-dalle
WORKDIR "$WORKDIR"

# user creation arguments
ARG user=dalle
ARG group=dalle
ARG uid=1000
ARG gid=1000

# create new user to run application
RUN groupadd --gid ${gid} ${group} && \
    useradd --create-home --home /home/${user}/ --uid ${uid} --gid ${gid} -s /bin/bash ${user}
RUN chown -R ${user}:${group} .
RUN chmod 755 ${WORKDIR}
USER ${user}

ENV PATH=/home/${user}/.local/bin:$PATH;
COPY requirements.txt "$WORKDIR"
RUN python -m venv venv
RUN pip install --no-cache-dir -r requirements.txt

COPY . "$WORKDIR"

# CMD bash setup.sh

CMD [ "python", "image_from_text.py", "--text='artificial intelligence'", "--torch", "--image_path='./generated'" ]
